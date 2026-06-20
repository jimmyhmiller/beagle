//! Inlining of non-escaping, single-expression local closures (slice 1).
//!
//! Targets the pattern
//! ```ignore
//! let f = fn(p0, p1, ...) { EXPR }   // single-expression body
//! ... f(a0, a1, ...) ...             // called, never escaping
//! ```
//! and rewrites each call `f(a0, ...)` to `EXPR` with each parameter `pi`
//! textually substituted by the corresponding argument `ai`, then deletes the
//! `let f = ...` binding. This removes BOTH the per-creation closure
//! allocation and the indirect call (see `bench_closure_probe.bg`: ~14x
//! between a closure-in-loop and the hand-inlined equivalent).
//!
//! ## Why each restriction is needed for soundness
//!
//! - **Captures are by value.** `make_closure` snapshots free variables into
//!   the closure object at creation time. So inlining (which reads the
//!   variable at *call* time) is only equivalent if no captured variable is
//!   mutated between the `let f` and the call. We bail if any identifier the
//!   body references (other than a parameter) is assigned anywhere in the
//!   region after the binding.
//! - **f must not escape.** The only sound use of `f` is a direct call
//!   `Call{name:"f"}` (the call name is a `String`, not an `Ast` child, so it
//!   is never an `Identifier("f")`). Any `Identifier("f")` in the region means
//!   `f` is passed/stored/returned/reassigned — bail.
//! - **Arguments must be pure** (a literal or a variable read). Direct
//!   substitution can duplicate or reorder an argument, so a side-effecting or
//!   non-idempotent argument would change semantics. Pure args are safe to
//!   duplicate and evaluate at the use site.
//! - **Single-expression body, fully-substitutable.** `substitute` returns
//!   `None` on any AST variant it does not fully traverse, which bails the
//!   whole inline — so we can never half-substitute a parameter and silently
//!   miss an occurrence.
//!
//! Conservative by construction: every uncertain case bails and leaves the
//! closure allocated (a missed optimization, never a miscompile).

#![allow(dead_code)]

use crate::ast::{Ast, Pattern};

/// Enabled by default; opt out with `BEAGLE_INLINE_CLOSURES=0`.
pub fn inline_enabled() -> bool {
    std::env::var("BEAGLE_INLINE_CLOSURES")
        .map(|v| v != "0")
        .unwrap_or(true)
}

fn dummy_tr() -> crate::ast::TokenRange {
    crate::ast::TokenRange { start: 0, end: 0 }
}

/// Entry point: inline eligible local closures inside every FUNCTION body
/// reachable from `ast`. Like scalar replacement, this only operates inside
/// closed function scopes — never on Program / namespace top-level bindings
/// (which persist across REPL lines and are visible to other functions, so
/// their escape can't be judged locally).
pub fn inline_in_ast(ast: &mut Ast) {
    match ast {
        Ast::Function { body, .. } => inline_in_body(body),
        Ast::Program { elements, .. } => {
            for el in elements.iter_mut() {
                inline_in_ast(el);
            }
        }
        _ => for_each_child_ast_mut(ast, &mut inline_in_ast),
    }
}

/// Inline eligible closures in a single statement list (a function body or a
/// nested block within one), recursing into nested blocks first.
fn inline_in_body(body: &mut Vec<Ast>) {
    // 1. Recurse into nested blocks (and inner closures' bodies) first.
    for stmt in body.iter_mut() {
        for_each_child_body(stmt, &mut inline_in_body);
    }
    // 2. Process this block left to right.
    let mut i = 0;
    while i < body.len() {
        if let Some(plan) = plan_inline(body, i) {
            // Replace every call site in the region after the binding.
            for stmt in body[i + 1..].iter_mut() {
                replace_calls(stmt, &plan);
            }
            // Drop the now-dead `let f = ...` binding.
            body.remove(i);
            // Do not advance i: the statement that followed now sits at i and
            // may itself be another inlinable closure binding.
        } else {
            i += 1;
        }
    }
}

/// A validated inlining plan for the closure bound at some index.
struct InlinePlan {
    /// Closure binding name (`f`).
    name: String,
    /// Parameter names in order.
    params: Vec<String>,
    /// The single body expression to substitute into.
    body_expr: Ast,
}

/// If `body[i]` binds an inlinable non-escaping closure, validate every
/// condition and return a plan; otherwise `None`.
fn plan_inline(body: &[Ast], i: usize) -> Option<InlinePlan> {
    let Ast::Let { pattern, value, .. } = &body[i] else {
        return None;
    };
    let Pattern::Identifier { name, .. } = pattern else {
        return None;
    };
    let Ast::Function {
        name: fn_name,
        args,
        rest_param,
        body: fn_body,
        ..
    } = value.as_ref()
    else {
        return None;
    };
    // No rest params; all params must be plain identifiers.
    if rest_param.is_some() {
        return None;
    }
    let mut params = Vec::with_capacity(args.len());
    for p in args {
        match p {
            Pattern::Identifier { name, .. } => params.push(name.clone()),
            _ => return None,
        }
    }
    // A named inner function can recurse via its own name — only inline
    // anonymous closures to keep that out of scope.
    if fn_name.is_some() {
        return None;
    }
    // Single-expression body, usable as a value expression.
    let expr = single_value_expr(fn_body)?;
    // Body must not reference the binding itself — neither as a bare
    // identifier nor as a recursive `Call{name:f}` (which substitution would
    // leave dangling after we delete the binding) — and must be fully
    // substitutable (no unhandled / shadowing-capable variants).
    if references_name(expr, name) || !is_substitutable(expr, &params) {
        return None;
    }

    let region = &body[i + 1..];
    // f must not escape (no non-call use of the name).
    if references_ident_in_region(region, name) {
        return None;
    }
    // Captures must not be mutated in the region (by-value capture semantics).
    let assigned = collect_assigned_idents(region);
    if expr_references_any_capture_assigned(expr, &params, &assigned) {
        return None;
    }
    // Every call site must be arity-matched with pure arguments.
    if !all_calls_inlinable(region, name, params.len()) {
        return None;
    }

    Some(InlinePlan {
        name: name.clone(),
        params,
        body_expr: expr.clone(),
    })
}

/// The single value expression of a closure body, if the body is exactly one
/// statement that is an expression (a trailing `return e` is unwrapped to
/// `e`). Statement-like single bodies (`let`, loops) are rejected.
fn single_value_expr(body: &[Ast]) -> Option<&Ast> {
    if body.len() != 1 {
        return None;
    }
    match &body[0] {
        Ast::Let { .. } | Ast::LetMut { .. } | Ast::LetDynamic { .. } => None,
        Ast::Return { value, .. } => Some(value.as_ref()),
        other => Some(other),
    }
}

/// Whether `node` references `name` anywhere as a bare identifier.
fn references_ident(node: &Ast, name: &str) -> bool {
    if matches!(node, Ast::Identifier(n, _) if n == name) {
        return true;
    }
    let mut found = false;
    for_each_child_ast(node, &mut |c| found = found || references_ident(c, name));
    found
}

fn references_ident_in_region(region: &[Ast], name: &str) -> bool {
    region.iter().any(|n| references_ident(n, name))
}

/// Whether `node` references `name` as a bare identifier OR as a direct call
/// `Call{name}`. Used for the non-recursion check on a closure body.
fn references_name(node: &Ast, name: &str) -> bool {
    if matches!(node, Ast::Identifier(n, _) if n == name) {
        return true;
    }
    if matches!(node, Ast::Call { name: n, .. } if n == name) {
        return true;
    }
    let mut found = false;
    for_each_child_ast(node, &mut |c| found = found || references_name(c, name));
    found
}

/// Whether `expr` can be cloned-and-substituted by `substitute` without ever
/// hitting an unhandled variant (which would risk a missed occurrence) — i.e.
/// `substitute` would not bail. Also rejects any nested binder that could
/// shadow a parameter name.
fn is_substitutable(expr: &Ast, params: &[String]) -> bool {
    // A successful trial substitution (identity args) proves traversability.
    let identity: Vec<Ast> = params
        .iter()
        .map(|p| Ast::Identifier(p.clone(), 0))
        .collect();
    substitute(expr, params, &identity).is_some()
}

/// Collect bare-identifier assignment targets (`n = e`) anywhere in `region`,
/// recursing through nested blocks. Over-collecting only forgoes inlining.
fn collect_assigned_idents(region: &[Ast]) -> Vec<String> {
    let mut out = Vec::new();
    for n in region {
        collect_assigned(n, &mut out);
    }
    out
}

fn collect_assigned(node: &Ast, out: &mut Vec<String>) {
    if let Ast::Assignment { name, .. } = node {
        if let Ast::Identifier(n, _) = name.as_ref() {
            if !out.iter().any(|x| x == n) {
                out.push(n.clone());
            }
        }
    }
    for_each_child_ast(node, &mut |c| collect_assigned(c, out));
}

/// Whether `expr` references any captured identifier (not a parameter) that is
/// in the `assigned` set — meaning a by-value capture would differ from the
/// inlined by-reference read.
fn expr_references_any_capture_assigned(
    expr: &Ast,
    params: &[String],
    assigned: &[String],
) -> bool {
    let mut bad = false;
    let check =
        |n: &str| -> bool { !params.iter().any(|p| p == n) && assigned.iter().any(|a| a == n) };
    fn walk(node: &Ast, check: &dyn Fn(&str) -> bool, bad: &mut bool) {
        if let Ast::Identifier(n, _) = node {
            if check(n) {
                *bad = true;
            }
        }
        for_each_child_ast(node, &mut |c| walk(c, check, bad));
    }
    walk(expr, &check, &mut bad);
    bad
}

/// Whether every `Call{name}` in `region` is arity-matched with pure args.
fn all_calls_inlinable(region: &[Ast], name: &str, arity: usize) -> bool {
    let mut ok = true;
    for n in region {
        check_calls(n, name, arity, &mut ok);
    }
    ok
}

fn check_calls(node: &Ast, name: &str, arity: usize, ok: &mut bool) {
    if let Ast::Call { name: cn, args, .. } = node {
        if cn == name && (args.len() != arity || !args.iter().all(is_pure_arg)) {
            *ok = false;
        }
    }
    for_each_child_ast(node, &mut |c| check_calls(c, name, arity, ok));
}

/// A pure argument: safe to duplicate / move to the use site (no side effects,
/// idempotent). Literals and plain variable reads.
fn is_pure_arg(a: &Ast) -> bool {
    matches!(
        a,
        Ast::IntegerLiteral(..)
            | Ast::FloatLiteral(..)
            | Ast::String(..)
            | Ast::Keyword(..)
            | Ast::True(..)
            | Ast::False(..)
            | Ast::Null(..)
            | Ast::Identifier(..)
    )
}

/// Replace every `Call{name == plan.name}` in `node` with the substituted
/// body expression. Recurses everywhere.
fn replace_calls(node: &mut Ast, plan: &InlinePlan) {
    // First recurse into children (handles nested calls in arguments etc.).
    for_each_child_ast_mut(node, &mut |c| replace_calls(c, plan));
    if let Ast::Call { name, args, .. } = node {
        if name == &plan.name && args.len() == plan.params.len() {
            // Guaranteed substitutable + pure args by plan_inline validation.
            if let Some(inlined) = substitute(&plan.body_expr, &plan.params, args) {
                *node = inlined;
            }
        }
    }
}

/// Clone `expr`, replacing each `Identifier(params[k])` with `args[k]`.
/// Returns `None` on any variant not fully handled (caller bails the inline).
fn substitute(expr: &Ast, params: &[String], args: &[Ast]) -> Option<Ast> {
    let sub = |e: &Ast| substitute(e, params, args);
    let sub_box = |e: &Ast| sub(e).map(Box::new);
    let sub_vec = |v: &[Ast]| -> Option<Vec<Ast>> { v.iter().map(|e| sub(e)).collect() };
    Some(match expr {
        Ast::Identifier(n, _) => {
            if let Some(k) = params.iter().position(|p| p == n) {
                args[k].clone()
            } else {
                expr.clone()
            }
        }
        Ast::IntegerLiteral(..)
        | Ast::FloatLiteral(..)
        | Ast::String(..)
        | Ast::Keyword(..)
        | Ast::True(..)
        | Ast::False(..)
        | Ast::Null(..) => expr.clone(),

        Ast::Add { left, right, .. } => Ast::Add {
            left: sub_box(left)?,
            right: sub_box(right)?,
            token_range: dummy_tr(),
        },
        Ast::Sub { left, right, .. } => Ast::Sub {
            left: sub_box(left)?,
            right: sub_box(right)?,
            token_range: dummy_tr(),
        },
        Ast::Mul { left, right, .. } => Ast::Mul {
            left: sub_box(left)?,
            right: sub_box(right)?,
            token_range: dummy_tr(),
        },
        Ast::Div { left, right, .. } => Ast::Div {
            left: sub_box(left)?,
            right: sub_box(right)?,
            token_range: dummy_tr(),
        },
        Ast::Modulo { left, right, .. } => Ast::Modulo {
            left: sub_box(left)?,
            right: sub_box(right)?,
            token_range: dummy_tr(),
        },
        Ast::Condition {
            operator,
            left,
            right,
            ..
        } => Ast::Condition {
            operator: *operator,
            left: sub_box(left)?,
            right: sub_box(right)?,
            token_range: dummy_tr(),
        },
        Ast::And { left, right, .. } => Ast::And {
            left: sub_box(left)?,
            right: sub_box(right)?,
            token_range: dummy_tr(),
        },
        Ast::Or { left, right, .. } => Ast::Or {
            left: sub_box(left)?,
            right: sub_box(right)?,
            token_range: dummy_tr(),
        },
        Ast::Not { expr: e, .. } => Ast::Not {
            expr: sub_box(e)?,
            token_range: dummy_tr(),
        },
        Ast::PropertyAccess {
            object, property, ..
        } => Ast::PropertyAccess {
            object: sub_box(object)?,
            // A property name is a static field identifier, not a variable —
            // never substitute it.
            property: Box::new((**property).clone()),
            token_range: dummy_tr(),
        },
        Ast::Call {
            name, args: cargs, ..
        } => Ast::Call {
            name: name.clone(),
            args: sub_vec(cargs)?,
            token_range: dummy_tr(),
        },
        Ast::CallExpr {
            callee,
            args: cargs,
            ..
        } => Ast::CallExpr {
            callee: sub_box(callee)?,
            args: sub_vec(cargs)?,
            token_range: dummy_tr(),
        },
        Ast::If {
            condition,
            then,
            else_,
            ..
        } => Ast::If {
            condition: sub_box(condition)?,
            then: sub_vec(then)?,
            else_: sub_vec(else_)?,
            token_range: dummy_tr(),
        },
        Ast::IndexOperator { array, index, .. } => Ast::IndexOperator {
            array: sub_box(array)?,
            index: sub_box(index)?,
            token_range: dummy_tr(),
        },
        Ast::Array { array, .. } => Ast::Array {
            array: sub_vec(array)?,
            token_range: dummy_tr(),
        },
        // Anything else (Let, Function, Match, Try, While, Loop, effects,
        // StructCreation, string interpolation, ...) is not traversed here —
        // bail so we never miss substituting a parameter occurrence.
        _ => return None,
    })
}

// ---- Shared shallow AST walkers (read-only + mutable). -------------------

fn for_each_child_ast(node: &Ast, f: &mut dyn FnMut(&Ast)) {
    match node {
        Ast::PropertyAccess {
            object, property, ..
        } => {
            f(object);
            f(property);
        }
        Ast::Assignment { name, value, .. } => {
            f(name);
            f(value);
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
        Ast::Let { value, .. } | Ast::LetMut { value, .. } | Ast::LetDynamic { value, .. } => {
            f(value)
        }
        Ast::If {
            condition,
            then,
            else_,
            ..
        } => {
            f(condition);
            then.iter().for_each(&mut *f);
            else_.iter().for_each(&mut *f);
        }
        Ast::While {
            condition, body, ..
        } => {
            f(condition);
            body.iter().for_each(&mut *f);
        }
        Ast::Loop { body, .. } => body.iter().for_each(&mut *f),
        Ast::Call { args, .. } => args.iter().for_each(&mut *f),
        Ast::CallExpr { callee, args, .. } => {
            f(callee);
            args.iter().for_each(&mut *f);
        }
        Ast::Return { value, .. } => f(value),
        Ast::Array { array, .. } => array.iter().for_each(&mut *f),
        Ast::IndexOperator { array, index, .. } => {
            f(array);
            f(index);
        }
        Ast::StructCreation { fields, spread, .. } => {
            fields.iter().for_each(|(_, e)| f(e));
            if let Some(s) = spread {
                f(s);
            }
        }
        Ast::EnumCreation { fields, .. } => {
            fields.iter().for_each(|(_, e)| f(e));
        }
        Ast::Function { body, .. } => body.iter().for_each(&mut *f),
        _ => {}
    }
}

fn for_each_child_ast_mut(node: &mut Ast, f: &mut dyn FnMut(&mut Ast)) {
    match node {
        Ast::PropertyAccess {
            object, property, ..
        } => {
            f(object);
            f(property);
        }
        Ast::Assignment { name, value, .. } => {
            f(name);
            f(value);
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
        Ast::Let { value, .. } | Ast::LetMut { value, .. } | Ast::LetDynamic { value, .. } => {
            f(value)
        }
        Ast::If {
            condition,
            then,
            else_,
            ..
        } => {
            f(condition);
            then.iter_mut().for_each(&mut *f);
            else_.iter_mut().for_each(&mut *f);
        }
        Ast::While {
            condition, body, ..
        } => {
            f(condition);
            body.iter_mut().for_each(&mut *f);
        }
        Ast::Loop { body, .. } => body.iter_mut().for_each(&mut *f),
        Ast::Call { args, .. } => args.iter_mut().for_each(&mut *f),
        Ast::CallExpr { callee, args, .. } => {
            f(callee);
            args.iter_mut().for_each(&mut *f);
        }
        Ast::Return { value, .. } => f(value),
        Ast::Array { array, .. } => array.iter_mut().for_each(&mut *f),
        Ast::IndexOperator { array, index, .. } => {
            f(array);
            f(index);
        }
        Ast::StructCreation { fields, spread, .. } => {
            fields.iter_mut().for_each(|(_, e)| f(e));
            if let Some(s) = spread {
                f(s);
            }
        }
        Ast::EnumCreation { fields, .. } => {
            fields.iter_mut().for_each(|(_, e)| f(e));
        }
        Ast::Function { body, .. } => body.iter_mut().for_each(&mut *f),
        _ => {}
    }
}

/// Call `f` on each nested statement-list of `stmt` (function/control blocks).
fn for_each_child_body(stmt: &mut Ast, f: &mut dyn FnMut(&mut Vec<Ast>)) {
    match stmt {
        Ast::Function { body, .. } => f(body),
        Ast::While { body, .. } => f(body),
        Ast::Loop { body, .. } => f(body),
        Ast::If { then, else_, .. } => {
            f(then);
            f(else_);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> Ast {
        Ast::Identifier(s.to_string(), 0)
    }
    fn int(n: i64) -> Ast {
        Ast::IntegerLiteral(n, 0)
    }
    fn tr() -> crate::ast::TokenRange {
        crate::ast::TokenRange { start: 0, end: 0 }
    }
    fn add(l: Ast, r: Ast) -> Ast {
        Ast::Add {
            left: Box::new(l),
            right: Box::new(r),
            token_range: tr(),
        }
    }
    fn closure(params: &[&str], body: Ast) -> Ast {
        Ast::Function {
            name: None,
            args: params
                .iter()
                .map(|p| Pattern::Identifier {
                    name: p.to_string(),
                    token_range: tr(),
                })
                .collect(),
            rest_param: None,
            body: vec![body],
            token_range: tr(),
            docstring: None,
        }
    }
    fn let_(name: &str, value: Ast) -> Ast {
        Ast::Let {
            pattern: Pattern::Identifier {
                name: name.to_string(),
                token_range: tr(),
            },
            value: Box::new(value),
            token_range: tr(),
            once: false,
        }
    }
    fn call(name: &str, args: Vec<Ast>) -> Ast {
        Ast::Call {
            name: name.to_string(),
            args,
            token_range: tr(),
        }
    }
    fn assign(lhs: Ast, rhs: Ast) -> Ast {
        Ast::Assignment {
            name: Box::new(lhs),
            value: Box::new(rhs),
            token_range: tr(),
        }
    }

    #[test]
    fn inlines_simple_non_escaping_closure() {
        // let f = fn(x){ x + base }; acc = acc + f(1)
        let mut body = vec![
            let_("f", closure(&["x"], add(id("x"), id("base")))),
            assign(id("acc"), add(id("acc"), call("f", vec![int(1)]))),
        ];
        inline_in_body(&mut body);
        // let removed -> single statement remains
        assert_eq!(body.len(), 1);
        // acc = acc + (1 + base)
        match &body[0] {
            Ast::Assignment { value, .. } => match value.as_ref() {
                Ast::Add { right, .. } => match right.as_ref() {
                    Ast::Add { left, right, .. } => {
                        assert!(matches!(left.as_ref(), Ast::IntegerLiteral(1, _)));
                        assert!(matches!(right.as_ref(), Ast::Identifier(n, _) if n == "base"));
                    }
                    _ => panic!("expected inlined (1 + base)"),
                },
                _ => panic!("expected add"),
            },
            _ => panic!("expected assignment"),
        }
    }

    #[test]
    fn does_not_inline_escaping_closure() {
        // let f = fn(x){x+1}; g(f)  — f passed as a value => escapes
        let mut body = vec![
            let_("f", closure(&["x"], add(id("x"), int(1)))),
            call("g", vec![id("f")]),
        ];
        let before = body.clone();
        inline_in_body(&mut body);
        assert_eq!(body.len(), before.len());
        assert!(matches!(&body[0], Ast::Let { .. }));
    }

    #[test]
    fn does_not_inline_when_capture_mutated() {
        // let f = fn(x){x+base}; base = 9; acc = f(1)
        // base is captured by value but mutated before the call => unsound.
        let mut body = vec![
            let_("f", closure(&["x"], add(id("x"), id("base")))),
            assign(id("base"), int(9)),
            assign(id("acc"), call("f", vec![int(1)])),
        ];
        inline_in_body(&mut body);
        // Unchanged: binding kept, call not inlined.
        assert!(matches!(&body[0], Ast::Let { .. }));
        assert_eq!(body.len(), 3);
    }

    #[test]
    fn does_not_inline_impure_arg() {
        // let f = fn(x){x+x}; acc = f(g())  — arg has a call (impure) and x
        // appears twice; duplicating g() would double its effects.
        let mut body = vec![
            let_("f", closure(&["x"], add(id("x"), id("x")))),
            assign(id("acc"), call("f", vec![call("g", vec![])])),
        ];
        inline_in_body(&mut body);
        assert!(matches!(&body[0], Ast::Let { .. }));
    }

    #[test]
    fn inlines_pure_variable_arg() {
        // let f = fn(x){ x * 2 }; acc = f(i)
        let mut body = vec![
            let_(
                "f",
                closure(
                    &["x"],
                    Ast::Mul {
                        left: Box::new(id("x")),
                        right: Box::new(int(2)),
                        token_range: tr(),
                    },
                ),
            ),
            assign(id("acc"), call("f", vec![id("i")])),
        ];
        inline_in_body(&mut body);
        assert_eq!(body.len(), 1);
        match &body[0] {
            Ast::Assignment { value, .. } => {
                assert!(matches!(value.as_ref(), Ast::Mul { .. }));
            }
            _ => panic!("expected assignment with inlined mul"),
        }
    }

    #[test]
    fn does_not_inline_recursive_closure() {
        // let f = fn(x){ f(x) }; f(1) — body references f (recursion)
        let mut body = vec![
            let_("f", closure(&["x"], call("f", vec![id("x")]))),
            call("f", vec![int(1)]),
        ];
        inline_in_body(&mut body);
        assert!(matches!(&body[0], Ast::Let { .. }));
    }
}
