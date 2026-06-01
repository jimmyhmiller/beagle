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

/// True if `v` escapes anywhere in `nodes` (used as anything other than a
/// `v.<ident>` field read).
pub fn var_escapes(nodes: &[Ast], v: &str) -> bool {
    nodes.iter().any(|n| escapes(n, v))
}

fn any(nodes: &[Ast], v: &str) -> bool {
    nodes.iter().any(|n| escapes(n, v))
}

fn escapes(node: &Ast, v: &str) -> bool {
    match node {
        // ---- The ONE allowed use: `v.<ident>` read. -------------------
        Ast::PropertyAccess {
            object, property, ..
        } if is_ident(object, v) && is_static_field(property) => false,

        // Any other property access: `v` could still appear inside the
        // object (e.g. `f(v).x`) — recurse into both halves.
        Ast::PropertyAccess {
            object, property, ..
        } => escapes(object, v) || escapes(property, v),

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
            target_touches_v || escapes(name, v) || escapes(value, v)
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
        | Ast::BitWiseXor { left, right, .. } => escapes(left, v) || escapes(right, v),

        Ast::Not { expr, .. } => escapes(expr, v),

        Ast::If {
            condition,
            then,
            else_,
            ..
        } => escapes(condition, v) || any(then, v) || any(else_, v),

        Ast::While {
            condition, body, ..
        } => escapes(condition, v) || any(body, v),

        Ast::Loop { body, .. } => any(body, v),

        // `Call.name` is a String (not an Ast); only args carry exprs. A
        // `v` in args means it's passed to a function → escapes.
        Ast::Call { args, .. } => any(args, v),
        Ast::CallExpr { callee, args, .. } => escapes(callee, v) || any(args, v),

        Ast::Return { value, .. } => escapes(value, v),

        Ast::Array { array, .. } => any(array, v),
        Ast::IndexOperator { array, index, .. } => escapes(array, v) || escapes(index, v),

        Ast::StructCreation { fields, spread, .. } => {
            fields.iter().any(|(_, e)| escapes(e, v))
                || spread.as_ref().map(|s| escapes(s, v)).unwrap_or(false)
        }

        // ---- Conservative default: any variant not handled above
        //      (Match, Try, For, MapLiteral, StringInterpolation, Let,
        //      closures, effects, ...) is treated as an ESCAPE. Sound —
        //      can only cost an optimization, never correctness. ---------
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Ast;

    fn id(s: &str) -> Ast {
        Ast::Identifier(s.to_string(), 0)
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
        assert!(!var_escapes(&body, "v"));
    }

    #[test]
    fn escapes_when_returned() {
        let body = vec![Ast::Return {
            value: Box::new(id("v")),
            token_range: tr(),
        }];
        assert!(var_escapes(&body, "v"));
    }

    #[test]
    fn escapes_when_passed_to_call() {
        let body = vec![Ast::Call {
            name: "f".to_string(),
            args: vec![id("v")],
            token_range: tr(),
        }];
        assert!(var_escapes(&body, "v"));
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
        assert!(var_escapes(&body, "v"));
    }

    #[test]
    fn escapes_on_field_write() {
        // v.x = 5
        let body = vec![Ast::Assignment {
            name: Box::new(field("v", "x")),
            value: Box::new(int(5)),
            token_range: tr(),
        }];
        assert!(var_escapes(&body, "v"));
    }

    #[test]
    fn escapes_on_reassign() {
        // v = 5
        let body = vec![Ast::Assignment {
            name: Box::new(id("v")),
            value: Box::new(int(5)),
            token_range: tr(),
        }];
        assert!(var_escapes(&body, "v"));
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
        assert!(!var_escapes(&body, "v"));
    }
}
