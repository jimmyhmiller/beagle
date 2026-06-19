//! AST-level loop-invariant hoisting (guard-and-hoist).
//!
//! A monomorphic hot loop like
//!
//! ```text
//! while i < n { acc = acc + base * base; i = i + 1 }
//! ```
//!
//! recomputes the loop-invariant `base * base` every iteration — a load, a
//! guard, and a multiply whose result never changes. Hoisting it out is worth
//! ~29% on such kernels (measured). The CFG-level passes can't: the checked
//! multiply is a *bailing* `InlineBranch` terminator, and hoisting a trapping
//! op into a plain preheader is unsound for a zero-trip loop (it would
//! bail/throw/allocate where the original never runs).
//!
//! This pass does it soundly at the AST level by **guard-and-hoist**:
//!
//! ```text
//! if i < n {                       // reached only when the loop runs >= 1 time
//!   let $liv$0 = base * base        // invariant computed once, safely
//!   while i < n { acc = acc + $liv$0; i = i + 1 }
//! }
//! ```
//!
//! The `if c` guard is the whole trick: a trapping/allocating invariant is now
//! evaluated only on a path where the loop body would have run at least once,
//! so it bails/throws exactly when the original's first iteration would — same
//! observable behaviour. The inner `while` is left intact, so `break`/`continue`
//! need no rewriting. The `if` returns the `while`'s value when `c` holds and
//! `null` otherwise, matching the original `while` (whose result starts null).
//!
//! ## Soundness rules (all conservative — when unsure, don't hoist)
//!
//! 1. **Simple condition only.** `c` must be a comparison / boolean combination
//!    of identifiers and literals (e.g. `i < n`). Then `c` is pure,
//!    non-trapping, and non-allocating, so the one extra evaluation the guard
//!    introduces is unobservable. Anything else (calls, arithmetic that could
//!    allocate) → skip the loop.
//! 2. **Invariant = not mutated, not loop-local.** A name is hoistable only if
//!    it is never an assignment target and never bound inside the loop body.
//!    The mutation scan **bails on any construct it doesn't fully model**
//!    (closures — which could mutate a captured var — `match`, `try`, effects,
//!    …); a bail means the loop is left untouched. Unknown ⇒ sound.
//! 3. **Pure arithmetic only.** Only subtrees built from arithmetic / bitwise /
//!    comparison / `not` over invariant identifiers and literals are hoisted —
//!    never calls, field reads, or indexing (which could have effects or read
//!    mutable state).
//! 4. **Unconditionally-executed positions only.** Hoisting descends only
//!    through eager operands and statement positions that run on *every*
//!    iteration. It stops at `if`/`&&`/`||` branches, nested loops, and closure
//!    bodies — a subexpression guarded by an inner conditional might never run,
//!    so hoisting it could trap where the original never would.
//!
//! ON by default; `BEAGLE_HOIST_LICM=0` opts out. Runs alongside the other AST
//! passes (escape analysis, closure inlining) before lowering.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::ast::{Ast, Pattern};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn fresh_name() -> String {
    // `$` cannot appear in user source identifiers, so these never collide.
    format!("$liv${}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

fn ident(name: &str) -> Ast {
    Ast::Identifier(name.to_string(), 0)
}

fn dummy_tr() -> crate::ast::TokenRange {
    crate::ast::TokenRange { start: 0, end: 0 }
}

/// Whether the pass is enabled. ON by default; `BEAGLE_HOIST_LICM=0` opts out.
pub fn hoist_enabled() -> bool {
    std::env::var("BEAGLE_HOIST_LICM")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Entry point: hoist loop invariants inside every function body reachable from
/// `ast`. Like the other AST passes, only function bodies (closed scopes) are
/// rewritten; top-level statements are descended for nested functions only.
pub fn hoist_in_ast(ast: &mut Ast) {
    process_node(ast);
}

/// Recurse to every body-bearing node, processing each statement list.
fn process_node(node: &mut Ast) {
    match node {
        Ast::Function { body, .. } => process_body(body),
        Ast::Program { elements, .. } => {
            for el in elements.iter_mut() {
                process_node(el);
            }
        }
        Ast::If {
            condition,
            then,
            else_,
            ..
        } => {
            process_node(condition);
            process_body(then);
            process_body(else_);
        }
        Ast::While {
            condition, body, ..
        } => {
            process_node(condition);
            process_body(body);
        }
        Ast::Loop { body, .. } => process_body(body),
        Ast::For {
            collection, body, ..
        } => {
            process_node(collection);
            process_body(body);
        }
        // Any other node: descend into expression children so nested functions
        // / loops (e.g. in a `let f = fn() {...}`) are still reached.
        other => for_each_child_mut(other, &mut process_node),
    }
}

/// Process a statement list: recurse into each element, then rewrite any
/// top-level `while` into a guard-and-hoist `if`.
fn process_body(body: &mut Vec<Ast>) {
    for el in body.iter_mut() {
        process_node(el);
    }
    let mut i = 0;
    while i < body.len() {
        if let Ast::While {
            condition,
            body: wbody,
            ..
        } = &body[i]
        {
            if let Some((lets, new_while)) = try_hoist(condition, wbody) {
                let mut then = lets;
                then.push(new_while);
                body[i] = Ast::If {
                    condition: condition.clone(),
                    then,
                    else_: vec![],
                    token_range: dummy_tr(),
                };
            }
        }
        i += 1;
    }
}

/// Try to hoist invariants out of `while condition { wbody }`. Returns the
/// `let` bindings to place before the loop and the rewritten `while`, or `None`
/// if nothing is hoistable (or the loop isn't safely analyzable).
fn try_hoist(condition: &Ast, wbody: &[Ast]) -> Option<(Vec<Ast>, Ast)> {
    if !is_simple_condition(condition) {
        return None;
    }
    let mut excluded = HashSet::new();
    for stmt in wbody {
        collect_excluded(stmt, &mut excluded)?; // bail (None) on anything unmodeled
    }

    let mut new_body: Vec<Ast> = wbody.to_vec();
    let mut hoists: Vec<(String, Ast)> = Vec::new();
    let mut dedup: Vec<(Ast, String)> = Vec::new();
    // `blocked` becomes true once anything that can throw or have a side effect
    // is evaluated. After that point nothing is hoisted: moving a throwing
    // invariant ahead of a preceding effect/throw would reorder observable
    // behaviour (which exception propagates, whether an effect ran first). It is
    // threaded across statements in evaluation order.
    let mut blocked = false;
    for stmt in new_body.iter_mut() {
        hoist_in_unconditional(stmt, &excluded, &mut hoists, &mut dedup, &mut blocked);
        if blocked {
            break;
        }
    }
    if hoists.is_empty() {
        return None;
    }

    let lets: Vec<Ast> = hoists
        .into_iter()
        .map(|(name, expr)| Ast::Let {
            pattern: Pattern::Identifier {
                name,
                token_range: dummy_tr(),
            },
            value: Box::new(expr),
            token_range: dummy_tr(),
            once: false,
        })
        .collect();
    let new_while = Ast::While {
        condition: Box::new(condition.clone()),
        body: new_body,
        token_range: dummy_tr(),
    };
    Some((lets, new_while))
}

// =========================================================================
// Condition purity
// =========================================================================

/// A "simple" condition: a comparison / boolean combination of identifiers and
/// literals. Such a condition is pure, non-trapping, and non-allocating, so the
/// extra evaluation introduced by the guard is unobservable.
fn is_simple_condition(node: &Ast) -> bool {
    match node {
        Ast::Identifier(..) | Ast::True(..) | Ast::False(..) => true,
        Ast::Condition { left, right, .. } => is_simple_operand(left) && is_simple_operand(right),
        Ast::And { left, right, .. } | Ast::Or { left, right, .. } => {
            is_simple_condition(left) && is_simple_condition(right)
        }
        Ast::Not { expr, .. } => is_simple_condition(expr),
        _ => false,
    }
}

fn is_simple_operand(node: &Ast) -> bool {
    matches!(
        node,
        Ast::Identifier(..)
            | Ast::IntegerLiteral(..)
            | Ast::FloatLiteral(..)
            | Ast::True(..)
            | Ast::False(..)
            | Ast::Null(..)
    )
}

// =========================================================================
// Mutation / binding scan (the "excluded" set)
// =========================================================================

/// Collect names that are NOT loop-invariant: assignment targets and names
/// bound inside the loop body. Returns `None` if the body contains any
/// construct this scan doesn't fully model (closures, `match`, `try`, effects,
/// …) — the caller then leaves the loop untouched. Conservative by design:
/// unknown ⇒ bail ⇒ no hoist.
fn collect_excluded(node: &Ast, out: &mut HashSet<String>) -> Option<()> {
    match node {
        // Leaves.
        Ast::IntegerLiteral(..)
        | Ast::FloatLiteral(..)
        | Ast::String(..)
        | Ast::Keyword(..)
        | Ast::Identifier(..)
        | Ast::True(..)
        | Ast::False(..)
        | Ast::Null(..)
        | Ast::Continue { .. } => Some(()),

        // Assignment: the target name becomes variant; recurse into both sides
        // (the target may be an index/property whose subexprs matter, and the
        // value may itself contain assignments).
        Ast::Assignment { name, value, .. } => {
            if let Ast::Identifier(n, _) = name.as_ref() {
                out.insert(n.clone());
            }
            collect_excluded(name, out)?;
            collect_excluded(value, out)?;
            Some(())
        }

        // Binders: names bound inside the loop are not available before it.
        Ast::Let { pattern, value, .. } | Ast::LetMut { pattern, value, .. } => {
            for n in pattern.binding_names() {
                out.insert(n);
            }
            collect_excluded(value, out)
        }

        // Eager binary / unary operators and boolean combinators.
        Ast::Add { left, right, .. }
        | Ast::Sub { left, right, .. }
        | Ast::Mul { left, right, .. }
        | Ast::Div { left, right, .. }
        | Ast::Modulo { left, right, .. }
        | Ast::Condition { left, right, .. }
        | Ast::And { left, right, .. }
        | Ast::Or { left, right, .. }
        | Ast::ShiftLeft { left, right, .. }
        | Ast::ShiftRight { left, right, .. }
        | Ast::ShiftRightZero { left, right, .. }
        | Ast::BitWiseAnd { left, right, .. }
        | Ast::BitWiseOr { left, right, .. }
        | Ast::BitWiseXor { left, right, .. } => {
            collect_excluded(left, out)?;
            collect_excluded(right, out)
        }
        Ast::Not { expr, .. } => collect_excluded(expr, out),

        // Calls / structured expressions: recurse into all children.
        Ast::Call { args, .. } => {
            for a in args {
                collect_excluded(a, out)?;
            }
            Some(())
        }
        Ast::CallExpr { callee, args, .. } => {
            collect_excluded(callee, out)?;
            for a in args {
                collect_excluded(a, out)?;
            }
            Some(())
        }
        Ast::If {
            condition,
            then,
            else_,
            ..
        } => {
            collect_excluded(condition, out)?;
            for s in then.iter().chain(else_.iter()) {
                collect_excluded(s, out)?;
            }
            Some(())
        }
        Ast::While {
            condition, body, ..
        } => {
            collect_excluded(condition, out)?;
            for s in body {
                collect_excluded(s, out)?;
            }
            Some(())
        }
        Ast::Loop { body, .. } => {
            for s in body {
                collect_excluded(s, out)?;
            }
            Some(())
        }
        Ast::For {
            binding,
            collection,
            body,
            ..
        } => {
            out.insert(binding.clone());
            collect_excluded(collection, out)?;
            for s in body {
                collect_excluded(s, out)?;
            }
            Some(())
        }
        Ast::Array { array, .. } => {
            for e in array {
                collect_excluded(e, out)?;
            }
            Some(())
        }
        Ast::IndexOperator { array, index, .. } => {
            collect_excluded(array, out)?;
            collect_excluded(index, out)
        }
        Ast::PropertyAccess { object, .. } => collect_excluded(object, out),
        Ast::StructCreation { fields, spread, .. } => {
            for (_, e) in fields {
                collect_excluded(e, out)?;
            }
            if let Some(s) = spread {
                collect_excluded(s, out)?;
            }
            Some(())
        }
        Ast::MapLiteral { pairs, .. } => {
            for (k, v) in pairs {
                collect_excluded(k, out)?;
                collect_excluded(v, out)?;
            }
            Some(())
        }
        Ast::SetLiteral { elements, .. } => {
            for e in elements {
                collect_excluded(e, out)?;
            }
            Some(())
        }
        Ast::StringInterpolation { parts, .. } => {
            for p in parts {
                if let crate::ast::StringInterpolationPart::Expression(e) = p {
                    collect_excluded(e, out)?;
                }
            }
            Some(())
        }
        Ast::Return { value, .. } | Ast::Break { value, .. } => collect_excluded(value, out),

        // Anything else (closures, match, try, effects, reset/shift, recurse,
        // enum creation, dynamic bindings, …) is not modeled — bail so the loop
        // is left untouched.
        _ => None,
    }
}

// =========================================================================
// Hoisting in unconditionally-executed positions
// =========================================================================

/// Walk the unconditionally-executed positions of `node` in evaluation order,
/// replacing each maximal invariant pure-arithmetic subtree with a reference to
/// a fresh hoisted temp — but only while `blocked` is false. `blocked` flips to
/// true the moment a node that can throw or have a side effect is evaluated, so
/// a (possibly-throwing) invariant is never hoisted ahead of a preceding
/// effect/throw. Recursion is strictly in evaluation order so the flag is set at
/// the right point.
fn hoist_in_unconditional(
    node: &mut Ast,
    excluded: &HashSet<String>,
    hoists: &mut Vec<(String, Ast)>,
    dedup: &mut Vec<(Ast, String)>,
    blocked: &mut bool,
) {
    if *blocked {
        return;
    }

    // Whole-node candidate: a computation whose every leaf is invariant. Hoisted
    // before any barrier is set, so it moves to the (guarded) preheader.
    if is_operator_root(node) && has_identifier(node) && fully_invariant_arith(node, excluded) {
        let name = match dedup.iter().find(|(e, _)| ast_eq(e, node)) {
            Some((_, n)) => n.clone(),
            None => {
                let n = fresh_name();
                dedup.push((node.clone(), n.clone()));
                hoists.push((n.clone(), node.clone()));
                n
            }
        };
        *node = ident(&name);
        return;
    }

    let mut recurse = |n: &mut Ast, blocked: &mut bool| {
        hoist_in_unconditional(n, excluded, hoists, dedup, blocked)
    };

    match node {
        // Non-invariant arithmetic/comparison: operands are evaluated (left then
        // right), then the operator itself may throw a type error / divide-by-
        // zero — so it becomes a barrier afterward.
        Ast::Add { left, right, .. }
        | Ast::Sub { left, right, .. }
        | Ast::Mul { left, right, .. }
        | Ast::Div { left, right, .. }
        | Ast::Modulo { left, right, .. }
        | Ast::Condition { left, right, .. }
        | Ast::ShiftLeft { left, right, .. }
        | Ast::ShiftRight { left, right, .. }
        | Ast::ShiftRightZero { left, right, .. }
        | Ast::BitWiseAnd { left, right, .. }
        | Ast::BitWiseOr { left, right, .. }
        | Ast::BitWiseXor { left, right, .. } => {
            recurse(left, blocked);
            recurse(right, blocked);
            *blocked = true;
        }
        // `&&` / `||`: only the left operand is unconditional; the right is
        // conditional, so block after the left.
        Ast::And { left, .. } | Ast::Or { left, .. } => {
            recurse(left, blocked);
            *blocked = true;
        }
        Ast::Not { expr, .. } => {
            recurse(expr, blocked);
            *blocked = true;
        }
        // Calls/reads: their sub-parts are evaluated first (and may yield
        // hoistable invariants), then the call/read can throw or have effects.
        Ast::Call { args, .. } => {
            for a in args.iter_mut() {
                recurse(a, blocked);
            }
            *blocked = true;
        }
        Ast::CallExpr { callee, args, .. } => {
            recurse(callee, blocked);
            for a in args.iter_mut() {
                recurse(a, blocked);
            }
            *blocked = true;
        }
        Ast::IndexOperator { array, index, .. } => {
            recurse(array, blocked);
            recurse(index, blocked);
            *blocked = true; // out-of-bounds read can throw
        }
        Ast::PropertyAccess { object, .. } => {
            recurse(object, blocked);
            *blocked = true; // missing-field read can throw
        }
        Ast::Assignment { name, value, .. } => {
            recurse(value, blocked);
            recurse(name, blocked);
            // A store to a local identifier is non-observable (discarded on
            // unwind) and can't throw; a store to a field/element can.
            if !matches!(name.as_ref(), Ast::Identifier(..)) {
                *blocked = true;
            }
        }
        Ast::Let { value, .. } | Ast::LetMut { value, .. } => recurse(value, blocked),
        Ast::Array { array, .. } => {
            for e in array.iter_mut() {
                recurse(e, blocked);
            }
            *blocked = true; // allocation; element evals may throw
        }
        Ast::StructCreation { fields, spread, .. } => {
            for (_, e) in fields.iter_mut() {
                recurse(e, blocked);
            }
            if let Some(s) = spread {
                recurse(s, blocked);
            }
            *blocked = true;
        }
        Ast::MapLiteral { pairs, .. } => {
            for (k, v) in pairs.iter_mut() {
                recurse(k, blocked);
                recurse(v, blocked);
            }
            *blocked = true;
        }
        Ast::SetLiteral { elements, .. } => {
            for e in elements.iter_mut() {
                recurse(e, blocked);
            }
            *blocked = true;
        }
        Ast::StringInterpolation { parts, .. } => {
            for p in parts.iter_mut() {
                if let crate::ast::StringInterpolationPart::Expression(e) = p {
                    recurse(e, blocked);
                }
            }
            *blocked = true;
        }
        Ast::Return { value, .. } | Ast::Break { value, .. } => {
            recurse(value, blocked);
            *blocked = true;
        }
        // Only the condition of an `if` is unconditional; the arms may run and
        // have effects we don't model — block after.
        Ast::If { condition, .. } => {
            recurse(condition, blocked);
            *blocked = true;
        }
        // Bare identifiers and literals are transparent (no throw, no effect).
        Ast::Identifier(..)
        | Ast::IntegerLiteral(..)
        | Ast::FloatLiteral(..)
        | Ast::String(..)
        | Ast::Keyword(..)
        | Ast::True(..)
        | Ast::False(..)
        | Ast::Null(..) => {}
        // Nested loops, closures, control flow, and everything else: stop.
        _ => *blocked = true,
    }
}

/// A node whose value is a pure-arithmetic function of invariant operands.
fn fully_invariant_arith(node: &Ast, excluded: &HashSet<String>) -> bool {
    match node {
        Ast::IntegerLiteral(..)
        | Ast::FloatLiteral(..)
        | Ast::True(..)
        | Ast::False(..)
        | Ast::Null(..) => true,
        Ast::Identifier(n, _) => !excluded.contains(n),
        Ast::Add { left, right, .. }
        | Ast::Sub { left, right, .. }
        | Ast::Mul { left, right, .. }
        | Ast::Div { left, right, .. }
        | Ast::Modulo { left, right, .. }
        | Ast::Condition { left, right, .. }
        | Ast::And { left, right, .. }
        | Ast::Or { left, right, .. }
        | Ast::ShiftLeft { left, right, .. }
        | Ast::ShiftRight { left, right, .. }
        | Ast::ShiftRightZero { left, right, .. }
        | Ast::BitWiseAnd { left, right, .. }
        | Ast::BitWiseOr { left, right, .. }
        | Ast::BitWiseXor { left, right, .. } => {
            fully_invariant_arith(left, excluded) && fully_invariant_arith(right, excluded)
        }
        Ast::Not { expr, .. } => fully_invariant_arith(expr, excluded),
        _ => false,
    }
}

/// Whether `node` is an arithmetic / boolean *operator* (so hoisting it does
/// real work — a bare identifier or literal is not worth a temp).
fn is_operator_root(node: &Ast) -> bool {
    matches!(
        node,
        Ast::Add { .. }
            | Ast::Sub { .. }
            | Ast::Mul { .. }
            | Ast::Div { .. }
            | Ast::Modulo { .. }
            | Ast::Condition { .. }
            | Ast::And { .. }
            | Ast::Or { .. }
            | Ast::ShiftLeft { .. }
            | Ast::ShiftRight { .. }
            | Ast::ShiftRightZero { .. }
            | Ast::BitWiseAnd { .. }
            | Ast::BitWiseOr { .. }
            | Ast::BitWiseXor { .. }
            | Ast::Not { .. }
    )
}

fn has_identifier(node: &Ast) -> bool {
    match node {
        Ast::Identifier(..) => true,
        Ast::Not { expr, .. } => has_identifier(expr),
        Ast::Add { left, right, .. }
        | Ast::Sub { left, right, .. }
        | Ast::Mul { left, right, .. }
        | Ast::Div { left, right, .. }
        | Ast::Modulo { left, right, .. }
        | Ast::Condition { left, right, .. }
        | Ast::And { left, right, .. }
        | Ast::Or { left, right, .. }
        | Ast::ShiftLeft { left, right, .. }
        | Ast::ShiftRight { left, right, .. }
        | Ast::ShiftRightZero { left, right, .. }
        | Ast::BitWiseAnd { left, right, .. }
        | Ast::BitWiseOr { left, right, .. }
        | Ast::BitWiseXor { left, right, .. } => has_identifier(left) || has_identifier(right),
        _ => false,
    }
}

/// Structural equality over the pure-arithmetic subset, ignoring token
/// positions. Used to dedup identical hoisted subexpressions.
fn ast_eq(a: &Ast, b: &Ast) -> bool {
    match (a, b) {
        (Ast::IntegerLiteral(x, _), Ast::IntegerLiteral(y, _)) => x == y,
        (Ast::FloatLiteral(x, _), Ast::FloatLiteral(y, _)) => x == y,
        (Ast::Identifier(x, _), Ast::Identifier(y, _)) => x == y,
        (Ast::True(..), Ast::True(..))
        | (Ast::False(..), Ast::False(..))
        | (Ast::Null(..), Ast::Null(..)) => true,
        (Ast::Not { expr: x, .. }, Ast::Not { expr: y, .. }) => ast_eq(x, y),
        (
            Ast::Add {
                left: la,
                right: ra,
                ..
            },
            Ast::Add {
                left: lb,
                right: rb,
                ..
            },
        )
        | (
            Ast::Sub {
                left: la,
                right: ra,
                ..
            },
            Ast::Sub {
                left: lb,
                right: rb,
                ..
            },
        )
        | (
            Ast::Mul {
                left: la,
                right: ra,
                ..
            },
            Ast::Mul {
                left: lb,
                right: rb,
                ..
            },
        )
        | (
            Ast::Div {
                left: la,
                right: ra,
                ..
            },
            Ast::Div {
                left: lb,
                right: rb,
                ..
            },
        )
        | (
            Ast::Modulo {
                left: la,
                right: ra,
                ..
            },
            Ast::Modulo {
                left: lb,
                right: rb,
                ..
            },
        )
        | (
            Ast::ShiftLeft {
                left: la,
                right: ra,
                ..
            },
            Ast::ShiftLeft {
                left: lb,
                right: rb,
                ..
            },
        )
        | (
            Ast::ShiftRight {
                left: la,
                right: ra,
                ..
            },
            Ast::ShiftRight {
                left: lb,
                right: rb,
                ..
            },
        )
        | (
            Ast::ShiftRightZero {
                left: la,
                right: ra,
                ..
            },
            Ast::ShiftRightZero {
                left: lb,
                right: rb,
                ..
            },
        )
        | (
            Ast::BitWiseAnd {
                left: la,
                right: ra,
                ..
            },
            Ast::BitWiseAnd {
                left: lb,
                right: rb,
                ..
            },
        )
        | (
            Ast::BitWiseOr {
                left: la,
                right: ra,
                ..
            },
            Ast::BitWiseOr {
                left: lb,
                right: rb,
                ..
            },
        )
        | (
            Ast::BitWiseXor {
                left: la,
                right: ra,
                ..
            },
            Ast::BitWiseXor {
                left: lb,
                right: rb,
                ..
            },
        )
        | (
            Ast::And {
                left: la,
                right: ra,
                ..
            },
            Ast::And {
                left: lb,
                right: rb,
                ..
            },
        )
        | (
            Ast::Or {
                left: la,
                right: ra,
                ..
            },
            Ast::Or {
                left: lb,
                right: rb,
                ..
            },
        ) => ast_eq(la, lb) && ast_eq(ra, rb),
        (
            Ast::Condition {
                operator: oa,
                left: la,
                right: ra,
                ..
            },
            Ast::Condition {
                operator: ob,
                left: lb,
                right: rb,
                ..
            },
        ) => oa == ob && ast_eq(la, lb) && ast_eq(ra, rb),
        _ => false,
    }
}

/// Minimal mutable child-walk for `process_node`'s generic case: reach nested
/// functions / loops inside expression-bearing nodes we don't special-case.
fn for_each_child_mut(node: &mut Ast, f: &mut dyn FnMut(&mut Ast)) {
    match node {
        Ast::Let { value, .. } | Ast::LetMut { value, .. } | Ast::Return { value, .. } => f(value),
        Ast::Assignment { name, value, .. } => {
            f(name);
            f(value);
        }
        Ast::Call { args, .. } => args.iter_mut().for_each(&mut *f),
        Ast::CallExpr { callee, args, .. } => {
            f(callee);
            args.iter_mut().for_each(&mut *f);
        }
        Ast::Add { left, right, .. }
        | Ast::Sub { left, right, .. }
        | Ast::Mul { left, right, .. }
        | Ast::Div { left, right, .. }
        | Ast::Modulo { left, right, .. }
        | Ast::Condition { left, right, .. }
        | Ast::And { left, right, .. }
        | Ast::Or { left, right, .. }
        | Ast::ShiftLeft { left, right, .. }
        | Ast::ShiftRight { left, right, .. }
        | Ast::ShiftRightZero { left, right, .. }
        | Ast::BitWiseAnd { left, right, .. }
        | Ast::BitWiseOr { left, right, .. }
        | Ast::BitWiseXor { left, right, .. } => {
            f(left);
            f(right);
        }
        Ast::Not { expr, .. } => f(expr),
        Ast::Array { array, .. } => array.iter_mut().for_each(&mut *f),
        Ast::IndexOperator { array, index, .. } => {
            f(array);
            f(index);
        }
        Ast::PropertyAccess {
            object, property, ..
        } => {
            f(object);
            f(property);
        }
        Ast::StructCreation { fields, spread, .. } => {
            fields.iter_mut().for_each(|(_, e)| f(e));
            if let Some(s) = spread {
                f(s);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    fn hoisted_dump(src: &str) -> String {
        let mut p = Parser::new("test".to_string(), src.to_string()).unwrap();
        let mut ast = p.parse().unwrap();
        hoist_in_ast(&mut ast);
        format!("{ast:?}")
    }

    fn parse_fn_body_has(src: &str, needle: &str) -> bool {
        hoisted_dump(src).contains(needle)
    }

    #[test]
    fn hoists_invariant_mul_out_of_while() {
        // `base * base` is invariant; after hoisting it should be bound in a
        // `$liv$` let and the while wrapped in an `if`.
        let src = r#"
namespace t
fn f(base, n) {
    let mut i = 0
    let mut acc = 0
    while i < n {
        acc = acc + base * base
        i = i + 1
    }
    acc
}
"#;
        let dump = hoisted_dump(src);
        assert!(dump.contains("$liv$"), "expected a hoisted temp binding");
        // The hoisted temp must be the product of the invariant operand.
        assert!(
            dump.matches("\"base\"").count() >= 2,
            "base should still be referenced inside the hoisted expression"
        );
    }

    #[test]
    fn does_not_hoist_variant_expression() {
        // `i * i` uses the loop counter `i`, which is assigned in the body —
        // not invariant, must not be hoisted.
        let src = r#"
namespace t
fn f(n) {
    let mut i = 0
    let mut acc = 0
    while i < n {
        acc = acc + i * i
        i = i + 1
    }
    acc
}
"#;
        assert!(
            !parse_fn_body_has(src, "$liv$"),
            "expression using the mutated loop counter must not be hoisted"
        );
    }

    #[test]
    fn does_not_hoist_with_nonsimple_condition() {
        // Condition contains a call → not a simple condition → skip the loop.
        let src = r#"
namespace t
fn f(base, xs) {
    let mut acc = 0
    while has_next(xs) {
        acc = acc + base * base
    }
    acc
}
"#;
        assert!(
            !parse_fn_body_has(src, "$liv$"),
            "loop with a non-simple condition must be skipped"
        );
    }
}
