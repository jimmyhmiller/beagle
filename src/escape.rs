//! Local escape analysis for scalar replacement of non-escaping structs.
//!
//! Given a variable `v` bound to a struct literal, decide whether `v`
//! **escapes** its scope — i.e. whether it is used in any way other than a
//! plain field read `v.<ident>`. If it does NOT escape, the allocation can
//! be scalar-replaced (each field becomes a local, `v.f` becomes that
//! local) with no heap object and therefore no GC involvement.
//!
//! **Soundness is conservative by construction:** the walk recurses into
//! every child of the variants it understands, and any variant it does NOT
//! explicitly handle falls through to `_ => true` (assume escape). So a
//! missed/forgotten variant can only make the analysis *over*-conservative
//! (skip an optimization), never unsound (miss a real escape).
//!
//! **Caller contract:** `v` must not be re-bound (shadowed) within the
//! analysed scope — the wiring that consumes this must verify that (or bail)
//! before trusting the result, since a re-`let v` would make later `v.f`
//! refer to a different value.

#![allow(dead_code)]

use crate::ast::Ast;

fn is_ident(a: &Ast, v: &str) -> bool {
    matches!(a, Ast::Identifier(n, _) if n == v)
}

fn is_static_field(a: &Ast) -> bool {
    matches!(a, Ast::Identifier(..))
}

/// True if `v` escapes anywhere in `nodes` — i.e. used as anything other
/// than a `v.<field>` READ where `field` is one of `fields` (the names the
/// struct literal actually provides). A read of a field NOT in `fields`
/// (e.g. a defaulted field) escapes, since scalar replacement has no value
/// to substitute.
pub fn var_escapes(nodes: &[Ast], v: &str, fields: &[String]) -> bool {
    nodes.iter().any(|n| escapes(n, v, fields))
}

fn any(nodes: &[Ast], v: &str, fields: &[String]) -> bool {
    nodes.iter().any(|n| escapes(n, v, fields))
}

/// A call name that mutates through a property lvalue (`primitive/set!`).
/// Its property-access argument is a WRITE target, not a read.
fn is_setter(name: &str) -> bool {
    name.ends_with("set!")
}

/// Does `v` appear anywhere in `node` (in any position)?
fn references_var(node: &Ast, v: &str) -> bool {
    if matches!(node, Ast::Identifier(n, _) if n == v) {
        return true;
    }
    let mut found = false;
    // Reuse the mutable child-walker via a cheap clone-free trick is awkward;
    // do a direct read-only structural check by cloning child refs is also
    // awkward — instead recurse through the same handled variants as escapes.
    for_each_child_ast(node, &mut |c| {
        found = found || references_var(c, v);
    });
    found
}

fn escapes(node: &Ast, v: &str, fields: &[String]) -> bool {
    match node {
        // A `set!`-style call whose args reference v is a mutation/escape.
        Ast::Call { name, args, .. } if is_setter(name) => {
            args.iter().any(|a| references_var(a, v))
        }

        // ---- The ONE allowed use: `v.<field>` read, field in literal. --
        Ast::PropertyAccess {
            object, property, ..
        } if is_ident(object, v)
            && matches!(property.as_ref(), Ast::Identifier(f, _) if fields.iter().any(|x| x == f)) =>
        {
            false
        }

        // Any other property access: `v` could still appear inside the
        // object (e.g. `f(v).x`) — recurse into both halves.
        Ast::PropertyAccess {
            object, property, ..
        } => escapes(object, v, fields) || escapes(property, v, fields),

        // ---- Assignment: writing through `v` (field write or rebind)
        //      is an escape; otherwise recurse target + value. ----------
        Ast::Assignment { name, value, .. } => {
            let target_touches_v = match name.as_ref() {
                // `v = e` — reassignment of v itself.
                Ast::Identifier(n, _) if n == v => true,
                // `v.f = e` — field write through v.
                Ast::PropertyAccess { object, .. } if is_ident(object, v) => true,
                _ => false,
            };
            target_touches_v || escapes(name, v, fields) || escapes(value, v, fields)
        }

        // ---- A bare use of `v` anywhere else = escape. ----------------
        Ast::Identifier(n, _) => n == v,

        // ---- Leaves: never contain v. ---------------------------------
        Ast::IntegerLiteral(..)
        | Ast::FloatLiteral(..)
        | Ast::String(..)
        | Ast::Keyword(..)
        | Ast::True(..)
        | Ast::False(..)
        | Ast::Null(..) => false,

        // ---- Composites: recurse into ALL children. -------------------
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
            escapes(left, v, fields) || escapes(right, v, fields)
        }

        Ast::Not { expr, .. } => escapes(expr, v, fields),

        Ast::If {
            condition,
            then,
            else_,
            ..
        } => escapes(condition, v, fields) || any(then, v, fields) || any(else_, v, fields),

        Ast::While {
            condition, body, ..
        } => escapes(condition, v, fields) || any(body, v, fields),

        Ast::Loop { body, .. } => any(body, v, fields),

        // `Call.name` is a String (not an Ast); only args carry exprs. A
        // `v` in args means it's passed to a function → escapes (bare v) or
        // is a read of v.field (allowed by the PropertyAccess arm).
        Ast::Call { args, .. } => any(args, v, fields),
        Ast::CallExpr { callee, args, .. } => escapes(callee, v, fields) || any(args, v, fields),

        Ast::Return { value, .. } => escapes(value, v, fields),

        Ast::Array { array, .. } => any(array, v, fields),
        Ast::IndexOperator { array, index, .. } => {
            escapes(array, v, fields) || escapes(index, v, fields)
        }

        Ast::StructCreation {
            fields: lit_fields,
            spread,
            ..
        } => {
            lit_fields.iter().any(|(_, e)| escapes(e, v, fields))
                || spread
                    .as_ref()
                    .map(|s| escapes(s, v, fields))
                    .unwrap_or(false)
        }

        // ---- Conservative default: any variant not handled above
        //      (Match, Try, For, MapLiteral, StringInterpolation, Let,
        //      closures, effects, ...) is treated as an ESCAPE. Sound —
        //      can only cost an optimization, never correctness. ---------
        _ => true,
    }
}

use crate::ast::Pattern;

fn dummy_tr() -> crate::ast::TokenRange {
    crate::ast::TokenRange { start: 0, end: 0 }
}

/// Synthetic local name for a scalar-replaced field. `$` cannot appear in
/// user source identifiers, so these never collide with user variables.
fn field_local(v: &str, field: &str) -> String {
    format!("{v}${field}")
}

/// Whether scalar replacement is enabled. ON by default (sound); opt out
/// with `BEAGLE_SCALAR_REPL=0` for A/B measurement and as an escape hatch.
pub fn scalar_repl_enabled() -> bool {
    std::env::var("BEAGLE_SCALAR_REPL")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Entry point: scalar-replace non-escaping structs throughout `ast`
/// (recursing into Program elements / function bodies).
pub fn scalar_replace_in_ast(ast: &mut Ast) {
    let mut wrapper = std::mem::replace(ast, Ast::Null(0));
    let mut buf = vec![wrapper];
    scalar_replace_structs(&mut buf);
    wrapper = buf.pop().unwrap();
    *ast = wrapper;
}

/// Scalar-replace non-escaping struct allocations in a statement list (and,
/// recursively, in nested bodies). For `let v = StructCreation{...}` where
/// `v` does not escape the rest of the block, replace the binding with one
/// `let v$field = fieldexpr` per field and rewrite every `v.field` read to
/// `v$field` — the heap object is never allocated.
///
/// Sound by construction: only fires when `var_escapes` (conservative) says
/// `v` is used solely as `v.<ident>` reads, so every such read is in a
/// variant the rewriter also handles; anything else makes `var_escapes`
/// return `true` and we leave the allocation untouched.
pub fn scalar_replace_structs(body: &mut Vec<Ast>) {
    // 1. Recurse into nested statement lists first (inner structs handled
    //    independently of this block's vars).
    for stmt in body.iter_mut() {
        for_each_child_body(stmt, &mut scalar_replace_structs);
    }
    // 2. Process this block left-to-right.
    let mut i = 0;
    while i < body.len() {
        if let Some((v, fields)) = replaceable_struct_let(body, i) {
            let fnames: Vec<String> = fields.iter().map(|(f, _)| f.clone()).collect();
            // Rewrite v.field -> v$field in the rest of the block.
            for stmt in body[i + 1..].iter_mut() {
                rewrite_field_reads(stmt, &v, &fnames);
            }
            // Replace the struct-let with one let per field.
            let lets: Vec<Ast> = fields
                .into_iter()
                .map(|(f, e)| Ast::Let {
                    pattern: Pattern::Identifier {
                        name: field_local(&v, &f),
                        token_range: dummy_tr(),
                    },
                    value: Box::new(e),
                    token_range: dummy_tr(),
                })
                .collect();
            let n = lets.len();
            body.splice(i..i + 1, lets);
            i += n;
        } else {
            i += 1;
        }
    }
}

/// If `body[i]` is `let v = StructCreation{spread: None}` with an identifier
/// pattern and `v` non-escaping over `body[i+1..]`, return `(v, fields)`.
fn replaceable_struct_let(body: &[Ast], i: usize) -> Option<(String, Vec<(String, Ast)>)> {
    let Ast::Let { pattern, value, .. } = &body[i] else {
        return None;
    };
    let Pattern::Identifier { name: v, .. } = pattern else {
        return None;
    };
    let Ast::StructCreation {
        fields,
        spread: None,
        ..
    } = value.as_ref()
    else {
        return None;
    };
    // v must not escape the rest of the block; only reads of fields the
    // literal actually provides are allowed.
    let field_names: Vec<String> = fields.iter().map(|(f, _)| f.clone()).collect();
    if var_escapes(&body[i + 1..], v, &field_names) {
        return None;
    }
    Some((v.clone(), fields.clone()))
}

/// Call `f` on each nested statement-list (Vec<Ast> body) of `stmt`. Covers
/// the control-flow nodes that hold blocks; struct-lets inside other nodes
/// (For/Match/closures) are simply not reached (left allocated — sound).
fn for_each_child_body(stmt: &mut Ast, f: &mut dyn FnMut(&mut Vec<Ast>)) {
    match stmt {
        Ast::Program { elements, .. } => f(elements),
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

/// Rewrite every `v.field` (field in `fnames`) READ to `Identifier(v$field)`,
/// recursing into the same variant set `escapes` understands. Field WRITES
/// and other uses can't occur here (else `var_escapes` would have bailed).
fn rewrite_field_reads(node: &mut Ast, v: &str, fnames: &[String]) {
    if let Ast::PropertyAccess {
        object, property, ..
    } = node
    {
        if is_ident(object, v) {
            if let Ast::Identifier(f, _) = property.as_ref() {
                if fnames.iter().any(|x| x == f) {
                    *node = Ast::Identifier(field_local(v, f), 0);
                    return;
                }
            }
        }
    }
    for_each_child_ast_mut(node, &mut |c| rewrite_field_reads(c, v, fnames));
}

/// Mutable visit of every direct child `Ast` of `node`. Covers the same
/// variants `escapes` recurses into, so every `v.field` the predicate
/// accepted is reached. Unhandled variants are no-ops (they can't contain
/// our `v` if `var_escapes` returned false).
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
        _ => {}
    }
}

/// Read-only mirror of `for_each_child_ast_mut`, for `references_var`.
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
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Ast;

    fn id(s: &str) -> Ast {
        Ast::Identifier(s.to_string(), 0)
    }
    fn fns() -> Vec<String> {
        vec!["x".to_string(), "y".to_string()]
    }
    fn int(n: i64) -> Ast {
        Ast::IntegerLiteral(n, 0)
    }
    fn tr() -> crate::ast::TokenRange {
        crate::ast::TokenRange { start: 0, end: 0 }
    }
    // `v.field` read
    fn field(obj: &str, f: &str) -> Ast {
        Ast::PropertyAccess {
            object: Box::new(id(obj)),
            property: Box::new(id(f)),
            token_range: tr(),
        }
    }

    #[test]
    fn non_escaping_field_reads() {
        // acc = v.x + v.y   (v used only as field reads)
        let body = vec![Ast::Assignment {
            name: Box::new(id("acc")),
            value: Box::new(Ast::Add {
                left: Box::new(field("v", "x")),
                right: Box::new(field("v", "y")),
                token_range: tr(),
            }),
            token_range: tr(),
        }];
        assert!(!var_escapes(&body, "v", &fns()));
    }

    #[test]
    fn escapes_when_returned() {
        let body = vec![Ast::Return {
            value: Box::new(id("v")),
            token_range: tr(),
        }];
        assert!(var_escapes(&body, "v", &fns()));
    }

    #[test]
    fn escapes_when_passed_to_call() {
        let body = vec![Ast::Call {
            name: "f".to_string(),
            args: vec![id("v")],
            token_range: tr(),
        }];
        assert!(var_escapes(&body, "v", &fns()));
    }

    #[test]
    fn escapes_when_aliased() {
        // let w = v   (v on the RHS of another binding) — Let is a
        // conservative-default variant, so escape.
        let body = vec![Ast::Let {
            pattern: crate::ast::Pattern::Identifier {
                name: "w".to_string(),
                token_range: tr(),
            },
            value: Box::new(id("v")),
            token_range: tr(),
        }];
        assert!(var_escapes(&body, "v", &fns()));
    }

    #[test]
    fn escapes_on_field_write() {
        // v.x = 5
        let body = vec![Ast::Assignment {
            name: Box::new(field("v", "x")),
            value: Box::new(int(5)),
            token_range: tr(),
        }];
        assert!(var_escapes(&body, "v", &fns()));
    }

    #[test]
    fn escapes_on_reassign() {
        // v = 5
        let body = vec![Ast::Assignment {
            name: Box::new(id("v")),
            value: Box::new(int(5)),
            token_range: tr(),
        }];
        assert!(var_escapes(&body, "v", &fns()));
    }

    fn struct_lit(fields: Vec<(&str, Ast)>) -> Ast {
        Ast::StructCreation {
            name: "V2".to_string(),
            fields: fields
                .into_iter()
                .map(|(n, e)| (n.to_string(), e))
                .collect(),
            spread: None,
            token_range: tr(),
        }
    }
    fn let_(name: &str, value: Ast) -> Ast {
        Ast::Let {
            pattern: crate::ast::Pattern::Identifier {
                name: name.to_string(),
                token_range: tr(),
            },
            value: Box::new(value),
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
    fn add(l: Ast, r: Ast) -> Ast {
        Ast::Add {
            left: Box::new(l),
            right: Box::new(r),
            token_range: tr(),
        }
    }

    #[test]
    fn rewrite_replaces_non_escaping_struct() {
        // let v = V2{x: 1, y: 2}; acc = v.x + v.y
        let mut body = vec![
            let_("v", struct_lit(vec![("x", int(1)), ("y", int(2))])),
            assign(id("acc"), add(field("v", "x"), field("v", "y"))),
        ];
        scalar_replace_structs(&mut body);
        // struct-let replaced by 2 field-lets => 3 statements total.
        assert_eq!(body.len(), 3);
        // first two are the field lets v$x = 1, v$y = 2
        match &body[0] {
            Ast::Let {
                pattern: crate::ast::Pattern::Identifier { name, .. },
                value,
                ..
            } => {
                assert_eq!(name, "v$x");
                assert!(matches!(value.as_ref(), Ast::IntegerLiteral(1, _)));
            }
            _ => panic!("expected let v$x"),
        }
        match &body[1] {
            Ast::Let {
                pattern: crate::ast::Pattern::Identifier { name, .. },
                ..
            } => {
                assert_eq!(name, "v$y");
            }
            _ => panic!("expected let v$y"),
        }
        // the assignment now reads v$x + v$y (no PropertyAccess left)
        match &body[2] {
            Ast::Assignment { value, .. } => match value.as_ref() {
                Ast::Add { left, right, .. } => {
                    assert!(matches!(left.as_ref(), Ast::Identifier(n, _) if n == "v$x"));
                    assert!(matches!(right.as_ref(), Ast::Identifier(n, _) if n == "v$y"));
                }
                _ => panic!("expected add of field locals"),
            },
            _ => panic!("expected assignment"),
        }
    }

    #[test]
    fn rewrite_leaves_escaping_struct_untouched() {
        // let v = V2{...}; return v   (v escapes) => unchanged
        let mut body = vec![
            let_("v", struct_lit(vec![("x", int(1)), ("y", int(2))])),
            Ast::Return {
                value: Box::new(id("v")),
                token_range: tr(),
            },
        ];
        let before = body.len();
        scalar_replace_structs(&mut body);
        assert_eq!(body.len(), before);
        assert!(
            matches!(&body[0], Ast::Let { value, .. } if matches!(value.as_ref(), Ast::StructCreation{..}))
        );
    }

    #[test]
    fn rewrite_inside_while_body() {
        // while c { let v = V2{x:1,y:2}; acc = acc + v.x + v.y }
        let inner = vec![
            let_("v", struct_lit(vec![("x", int(1)), ("y", int(2))])),
            assign(
                id("acc"),
                add(add(id("acc"), field("v", "x")), field("v", "y")),
            ),
        ];
        let mut body = vec![Ast::While {
            condition: Box::new(id("c")),
            body: inner,
            token_range: tr(),
        }];
        scalar_replace_structs(&mut body);
        if let Ast::While { body: inner, .. } = &body[0] {
            assert_eq!(inner.len(), 3); // 2 field lets + assignment
            assert!(!inner.iter().any(|s| matches!(s, Ast::Let { value, .. } if matches!(value.as_ref(), Ast::StructCreation{..}))));
        } else {
            panic!("while gone");
        }
    }

    #[test]
    fn field_reads_in_loop_are_safe() {
        // while cond { acc = acc + v.x }
        let body = vec![Ast::While {
            condition: Box::new(id("cond")),
            body: vec![Ast::Assignment {
                name: Box::new(id("acc")),
                value: Box::new(Ast::Add {
                    left: Box::new(id("acc")),
                    right: Box::new(field("v", "x")),
                    token_range: tr(),
                }),
                token_range: tr(),
            }],
            token_range: tr(),
        }];
        assert!(!var_escapes(&body, "v", &fns()));
    }
}
