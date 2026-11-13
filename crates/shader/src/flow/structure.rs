//! Structurisation: turns a CFG into structured IR by goto elimination (Erosa and Hendren),
//! then translates every block in program order and builds the syntax list.
//!
//! Every CFG edge starts as a goto in one flat list of labelled blocks. Gotos are moved
//! outward, lifted and moved inward until each is a sibling of its label, then replaced by an
//! `if` (forward) or a `do-while` (backward). Goto variables carry the conditions across
//! the moves.

use rustc_hash::FxHashMap;

use super::cfg::{BlockId as FlowBlockId, Cfg, EndClass, FunctionId};
use crate::TranslateError;
use crate::ir::{BlockId, Condition, Emitter, ICompare, Opcode, Program, Reg, SyntaxNode, Value};

type StmtId = u32;
type ExprId = u32;
const NIL: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
enum Expr {
    Identity(Condition),
    Not(ExprId),
    Or(ExprId, ExprId),
    Variable(u32),
    IndirectBranchCond(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Code { begin: u32, end: u32 },
    Goto { cond: ExprId, label: StmtId },
    Label { id: u32 },
    If { cond: ExprId },
    Loop { cond: ExprId },
    Break { cond: ExprId },
    Return,
    Kill,
    Unreachable,
    Function,
    SetVariable { id: u32, op: ExprId },
    SetIndirectBranchVariable { reg: Reg, offset: i32 },
}

impl Kind {
    fn has_children(self) -> bool {
        matches!(self, Kind::If { .. } | Kind::Loop { .. } | Kind::Function)
    }
}

#[derive(Clone, Copy, Debug)]
struct Stmt {
    kind: Kind,
    up: StmtId,
    prev: StmtId,
    next: StmtId,
    first: StmtId,
    last: StmtId,
}

/// The statement tree: every list is doubly linked through the arena so splices are O(length
/// of the moved range) at worst and inserts and erases are O(1).
struct Tree {
    stmts: Vec<Stmt>,
    exprs: Vec<Expr>,
}

const ROOT: StmtId = 0;

impl Tree {
    fn new() -> Tree {
        Tree {
            stmts: vec![Stmt {
                kind: Kind::Function,
                up: NIL,
                prev: NIL,
                next: NIL,
                first: NIL,
                last: NIL,
            }],
            exprs: Vec::new(),
        }
    }

    fn expr(&mut self, e: Expr) -> ExprId {
        self.exprs.push(e);
        (self.exprs.len() - 1) as ExprId
    }

    fn s(&self, id: StmtId) -> &Stmt {
        &self.stmts[id as usize]
    }

    fn sm(&mut self, id: StmtId) -> &mut Stmt {
        &mut self.stmts[id as usize]
    }

    fn create(&mut self, kind: Kind, up: StmtId) -> StmtId {
        self.stmts.push(Stmt {
            kind,
            up,
            prev: NIL,
            next: NIL,
            first: NIL,
            last: NIL,
        });
        (self.stmts.len() - 1) as StmtId
    }

    /// Links detached `stmt` into `parent`'s list before `before` (or at the end for NIL).
    fn insert(&mut self, parent: StmtId, before: StmtId, stmt: StmtId) -> StmtId {
        let prev = if before == NIL {
            self.s(parent).last
        } else {
            self.s(before).prev
        };
        {
            let s = self.sm(stmt);
            s.up = parent;
            s.prev = prev;
            s.next = before;
        }
        if prev == NIL {
            self.sm(parent).first = stmt;
        } else {
            self.sm(prev).next = stmt;
        }
        if before == NIL {
            self.sm(parent).last = stmt;
        } else {
            self.sm(before).prev = stmt;
        }
        stmt
    }

    fn push_front(&mut self, parent: StmtId, stmt: StmtId) -> StmtId {
        let first = self.s(parent).first;
        self.insert(parent, first, stmt)
    }

    fn push_back(&mut self, parent: StmtId, stmt: StmtId) -> StmtId {
        self.insert(parent, NIL, stmt)
    }

    fn erase(&mut self, stmt: StmtId) {
        let Stmt { up, prev, next, .. } = *self.s(stmt);
        if prev == NIL {
            self.sm(up).first = next;
        } else {
            self.sm(prev).next = next;
        }
        if next == NIL {
            self.sm(up).last = prev;
        } else {
            self.sm(next).prev = prev;
        }
        let s = self.sm(stmt);
        s.prev = NIL;
        s.next = NIL;
    }

    /// Moves `[first, end)` of `from`'s list (end NIL = to the end) into the empty list of
    /// `to`, updating parents.
    fn splice(&mut self, to: StmtId, from: StmtId, first: StmtId, end: StmtId) {
        if first == end {
            return;
        }
        let last = if end == NIL {
            self.s(from).last
        } else {
            self.s(end).prev
        };
        let before = self.s(first).prev;
        if before == NIL {
            self.sm(from).first = end;
        } else {
            self.sm(before).next = end;
        }
        if end == NIL {
            self.sm(from).last = before;
        } else {
            self.sm(end).prev = before;
        }
        self.sm(first).prev = NIL;
        self.sm(last).next = NIL;
        self.sm(to).first = first;
        self.sm(to).last = last;
        let mut cur = first;
        while cur != NIL {
            self.sm(cur).up = to;
            cur = self.s(cur).next;
        }
    }

    fn level(&self, mut stmt: StmtId) -> usize {
        let mut level = 0;
        while self.s(stmt).up != NIL {
            level += 1;
            stmt = self.s(stmt).up;
        }
        level
    }

    fn is_directly_related(&self, goto_stmt: StmtId, label: StmtId) -> bool {
        let (gl, ll) = (self.level(goto_stmt), self.level(label));
        let (min, mut max, min_level, mut max_level) = if ll < gl {
            (label, goto_stmt, ll, gl)
        } else {
            (goto_stmt, label, gl, ll)
        };
        while max_level > min_level {
            max_level -= 1;
            max = self.s(max).up;
        }
        self.s(min).up == self.s(max).up
    }

    fn is_indirectly_related(&self, goto_stmt: StmtId, label: StmtId) -> bool {
        self.s(goto_stmt).up != self.s(label).up && !self.is_directly_related(goto_stmt, label)
    }

    /// The ancestor of `nephew` that shares `uncle`'s parent.
    fn sibling_from_nephew(&self, uncle: StmtId, nephew: StmtId) -> StmtId {
        let parent = self.s(uncle).up;
        let mut it = nephew;
        while self.s(it).up != parent {
            it = self.s(it).up;
        }
        it
    }

    /// True when `left` comes before `right` among siblings. Walks forward from both at once so
    /// the cost is the shorter of the two walks.
    fn are_ordered(&self, left: StmtId, right: StmtId) -> bool {
        if left == right {
            return false;
        }
        let (mut l, mut r) = (left, right);
        loop {
            r = self.s(r).next;
            if r == left {
                return false;
            }
            if r == NIL {
                return true;
            }
            l = self.s(l).next;
            if l == right {
                return true;
            }
            if l == NIL {
                return false;
            }
        }
    }

    fn needs_lift(&self, goto_stmt: StmtId, label: StmtId) -> bool {
        let sibling = self.sibling_from_nephew(goto_stmt, label);
        self.are_ordered(sibling, goto_stmt)
    }

    fn goto_parts(&self, goto_stmt: StmtId) -> (ExprId, StmtId) {
        match self.s(goto_stmt).kind {
            Kind::Goto { cond, label } => (cond, label),
            _ => unreachable!("statement is not a goto"),
        }
    }

    fn label_id(&self, label: StmtId) -> u32 {
        match self.s(label).kind {
            Kind::Label { id } => id,
            _ => unreachable!("statement is not a label"),
        }
    }

    /// The first code statement after `stmt` before any nested construct.
    fn try_find_forward_block(&self, stmt: StmtId) -> Option<StmtId> {
        let mut node = self.s(stmt).next;
        while node != NIL && !self.s(node).kind.has_children() {
            if matches!(self.s(node).kind, Kind::Code { .. }) {
                return Some(node);
            }
            node = self.s(node).next;
        }
        None
    }
}

struct GotoPass<'a> {
    tree: Tree,
    cfg: &'a Cfg,
    exit_label: Option<StmtId>,
    label_id: u32,
    false_expr: ExprId,
    true_expr: ExprId,
}

impl<'a> GotoPass<'a> {
    fn run(cfg: &'a Cfg, converge_exits: bool) -> Result<Tree, TranslateError> {
        let mut tree = Tree::new();
        let false_expr = tree.expr(Expr::Identity(Condition::FALSE));
        let true_expr = tree.expr(Expr::Identity(Condition::TRUE));
        let mut pass = GotoPass {
            tree,
            cfg,
            exit_label: None,
            label_id: 0,
            false_expr,
            true_expr,
        };
        let mut gotos = Vec::new();
        let mut insert_point = NIL;
        if converge_exits {
            let label = pass.tree.create(Kind::Label { id: 0 }, ROOT);
            pass.label_id = 1;
            pass.tree.push_back(ROOT, label);
            let ret = pass.tree.create(Kind::Return, ROOT);
            pass.tree.push_back(ROOT, ret);
            pass.exit_label = Some(label);
            insert_point = label;
        }
        pass.build(0, &mut gotos, insert_point, None)?;
        for &goto_stmt in gotos.iter().rev() {
            pass.remove_goto(goto_stmt)?;
        }
        pass.reset_goto_variables();
        Ok(pass.tree)
    }

    fn build(
        &mut self,
        function: FunctionId,
        gotos: &mut Vec<StmtId>,
        insert_point: StmtId,
        return_label: Option<StmtId>,
    ) -> Result<(), TranslateError> {
        let cfg = self.cfg;
        let blocks: Vec<FlowBlockId> = cfg.functions[function].blocks().collect();
        let mut labels: FxHashMap<FlowBlockId, StmtId> = FxHashMap::default();
        labels.reserve(blocks.len());
        for &block in &blocks {
            let label = self.tree.create(Kind::Label { id: self.label_id }, ROOT);
            self.tree.insert(ROOT, insert_point, label);
            labels.insert(block, label);
            self.label_id += 1;
        }
        let label_of = |block: Option<FlowBlockId>| -> Result<StmtId, TranslateError> {
            block
                .and_then(|b| labels.get(&b).copied())
                .ok_or(TranslateError::Unstructured { address: 0 })
        };
        for &block_id in &blocks {
            let block = cfg.block(block_id);
            let label = labels[&block_id];
            let ip = self.tree.s(label).next;
            let id = self.tree.label_id(label);
            let reset = self.tree.create(
                Kind::SetVariable {
                    id,
                    op: self.false_expr,
                },
                ROOT,
            );
            self.tree.insert(ROOT, ip, reset);
            let code = self.tree.create(
                Kind::Code {
                    begin: block.begin,
                    end: block.end,
                },
                ROOT,
            );
            self.tree.insert(ROOT, ip, code);
            match block.end_class {
                EndClass::Branch => {
                    if block.cond.is_true() {
                        let target = label_of(block.branch_true)?;
                        gotos.push(self.goto(ip, self.true_expr, target));
                    } else if block.cond.is_false() {
                        let target = label_of(block.branch_false)?;
                        gotos.push(self.goto(ip, self.true_expr, target));
                    } else {
                        let t = label_of(block.branch_true)?;
                        let f = label_of(block.branch_false)?;
                        let cond = self.tree.expr(Expr::Identity(block.cond));
                        gotos.push(self.goto(ip, cond, t));
                        gotos.push(self.goto(ip, self.true_expr, f));
                    }
                }
                EndClass::IndirectBranch => {
                    let set = self.tree.create(
                        Kind::SetIndirectBranchVariable {
                            reg: block.branch_reg,
                            offset: block.branch_offset,
                        },
                        ROOT,
                    );
                    self.tree.insert(ROOT, ip, set);
                    for &(target, address) in &block.indirect_branches {
                        let cond = self.tree.expr(Expr::IndirectBranchCond(address));
                        let target = label_of(Some(target))?;
                        gotos.push(self.goto(ip, cond, target));
                    }
                    let unreachable = self.tree.create(Kind::Unreachable, ROOT);
                    self.tree.insert(ROOT, ip, unreachable);
                }
                EndClass::Call => {
                    let ret = label_of(block.return_block)?;
                    self.build(block.function_call, gotos, ip, Some(ret))?;
                }
                EndClass::Exit => match self.exit_label {
                    Some(exit) => gotos.push(self.goto(ip, self.true_expr, exit)),
                    None => {
                        let ret = self.tree.create(Kind::Return, ROOT);
                        self.tree.insert(ROOT, ip, ret);
                    }
                },
                EndClass::Return => {
                    let target = return_label.ok_or(TranslateError::Unstructured {
                        address: u64::from(block.begin),
                    })?;
                    let cond = self.tree.expr(Expr::Identity(block.cond));
                    gotos.push(self.goto(ip, cond, target));
                }
                EndClass::Kill => {
                    let kill = self.tree.create(Kind::Kill, ROOT);
                    self.tree.insert(ROOT, ip, kill);
                }
            }
        }
        Ok(())
    }

    /// Sets every goto variable that is read to false just before its first use, in the
    /// innermost list holding all its uses. Without gotos the tree enters a list only at its
    /// front, so that statement dominates every use, and SSA reads of the variable stop there
    /// instead of walking back to the entry.
    fn reset_goto_variables(&mut self) {
        let vars = self.label_id as usize;
        let mut first = vec![NIL; vars];
        let mut last = vec![NIL; vars];
        let mut read = vec![false; vars];
        let mut depth = vec![0u32; self.tree.stmts.len()];
        let mut exprs = Vec::new();
        let mut stack = vec![(self.tree.s(ROOT).first, 1u32)];
        while let Some((stmt, level)) = stack.pop() {
            if stmt == NIL {
                continue;
            }
            stack.push((self.tree.s(stmt).next, level));
            depth[stmt as usize] = level;
            let kind = self.tree.s(stmt).kind;
            match kind {
                Kind::SetVariable { id, op } => {
                    let id = id as usize;
                    if first[id] == NIL {
                        first[id] = stmt;
                    }
                    last[id] = stmt;
                    exprs.push(op);
                }
                Kind::If { cond } | Kind::Loop { cond } | Kind::Break { cond } => exprs.push(cond),
                _ => {}
            }
            while let Some(expr) = exprs.pop() {
                match self.tree.exprs[expr as usize] {
                    Expr::Variable(id) => {
                        let id = id as usize;
                        if first[id] == NIL {
                            first[id] = stmt;
                        }
                        last[id] = stmt;
                        read[id] = true;
                    }
                    Expr::Not(e) => exprs.push(e),
                    Expr::Or(a, b) => exprs.extend([a, b]),
                    Expr::Identity(_) | Expr::IndirectBranchCond(_) => {}
                }
            }
            if kind.has_children() {
                stack.push((self.tree.s(stmt).first, level + 1));
            }
        }
        for id in 0..vars {
            if !read[id] {
                continue;
            }
            let (mut a, mut b) = (first[id], last[id]);
            while depth[b as usize] > depth[a as usize] {
                b = self.tree.s(b).up;
            }
            while depth[a as usize] > depth[b as usize] {
                a = self.tree.s(a).up;
            }
            while self.tree.s(a).up != self.tree.s(b).up {
                a = self.tree.s(a).up;
                b = self.tree.s(b).up;
            }
            let parent = self.tree.s(a).up;
            let reset = self.tree.create(
                Kind::SetVariable {
                    id: id as u32,
                    op: self.false_expr,
                },
                parent,
            );
            self.tree.insert(parent, a, reset);
        }
    }

    fn goto(&mut self, before: StmtId, cond: ExprId, label: StmtId) -> StmtId {
        let stmt = self.tree.create(Kind::Goto { cond, label }, ROOT);
        self.tree.insert(ROOT, before, stmt)
    }

    fn remove_goto(&mut self, mut goto_stmt: StmtId) -> Result<(), TranslateError> {
        let (_, label) = self.tree.goto_parts(goto_stmt);
        if self.tree.is_indirectly_related(goto_stmt, label) {
            while !self.tree.is_directly_related(goto_stmt, label) {
                goto_stmt = self.move_outward(goto_stmt)?;
            }
        }
        if self.tree.is_directly_related(goto_stmt, label) {
            let label_level = self.tree.level(label);
            let mut goto_level = self.tree.level(goto_stmt);
            if goto_level > label_level {
                while goto_level > label_level {
                    goto_stmt = self.move_outward(goto_stmt)?;
                    goto_level -= 1;
                }
            } else {
                if self.tree.needs_lift(goto_stmt, label) {
                    goto_stmt = self.lift(goto_stmt)?;
                }
                while goto_level < label_level {
                    goto_stmt = self.move_inward(goto_stmt)?;
                    goto_level += 1;
                }
            }
        }
        if self.tree.s(goto_stmt).next == label {
            self.tree.erase(goto_stmt);
        } else if self.tree.are_ordered(goto_stmt, label) {
            self.eliminate_as_conditional(goto_stmt, label);
        } else {
            self.eliminate_as_loop(goto_stmt, label);
        }
        Ok(())
    }

    fn eliminate_as_conditional(&mut self, goto_stmt: StmtId, label: StmtId) {
        let parent = self.tree.s(goto_stmt).up;
        let (cond, _) = self.tree.goto_parts(goto_stmt);
        let not = self.tree.expr(Expr::Not(cond));
        let if_stmt = self.tree.create(Kind::If { cond: not }, parent);
        let first = self.tree.s(goto_stmt).next;
        self.tree.splice(if_stmt, parent, first, label);
        self.tree.insert(parent, goto_stmt, if_stmt);
        self.tree.erase(goto_stmt);
    }

    fn eliminate_as_loop(&mut self, goto_stmt: StmtId, label: StmtId) {
        let parent = self.tree.s(goto_stmt).up;
        let (cond, _) = self.tree.goto_parts(goto_stmt);
        let loop_stmt = self.tree.create(Kind::Loop { cond }, parent);
        self.tree.splice(loop_stmt, parent, label, goto_stmt);
        self.tree.insert(parent, goto_stmt, loop_stmt);
        self.tree.erase(goto_stmt);
    }

    fn move_outward(&mut self, goto_stmt: StmtId) -> Result<StmtId, TranslateError> {
        let parent = self.tree.s(goto_stmt).up;
        match self.tree.s(parent).kind {
            Kind::If { .. } => Ok(self.move_outward_if(goto_stmt)),
            Kind::Loop { .. } => Ok(self.move_outward_loop(goto_stmt)),
            _ => Err(TranslateError::Unstructured { address: 0 }),
        }
    }

    fn move_inward(&mut self, goto_stmt: StmtId) -> Result<StmtId, TranslateError> {
        let parent = self.tree.s(goto_stmt).up;
        let (goto_cond, label) = self.tree.goto_parts(goto_stmt);
        let nested = self.tree.sibling_from_nephew(goto_stmt, label);
        let label_id = self.tree.label_id(label);
        let set_var = self.tree.create(
            Kind::SetVariable {
                id: label_id,
                op: goto_cond,
            },
            parent,
        );
        self.tree.insert(parent, goto_stmt, set_var);
        let variable = self.tree.expr(Expr::Variable(label_id));
        let neg_var = self.tree.expr(Expr::Not(variable));
        let first = self.tree.s(goto_stmt).next;
        if first != nested {
            let if_stmt = self.tree.create(Kind::If { cond: neg_var }, parent);
            self.tree.splice(if_stmt, parent, first, nested);
            self.tree.insert(parent, goto_stmt, if_stmt);
        }
        self.tree.erase(goto_stmt);
        match self.tree.s(nested).kind {
            Kind::If { cond } => {
                let or = self.tree.expr(Expr::Or(variable, cond));
                self.tree.sm(nested).kind = Kind::If { cond: or };
            }
            Kind::Loop { .. } => {}
            _ => return Err(TranslateError::Unstructured { address: 0 }),
        }
        let new_goto = self.tree.create(
            Kind::Goto {
                cond: variable,
                label,
            },
            nested,
        );
        Ok(self.tree.push_front(nested, new_goto))
    }

    fn lift(&mut self, goto_stmt: StmtId) -> Result<StmtId, TranslateError> {
        let parent = self.tree.s(goto_stmt).up;
        let (goto_cond, label) = self.tree.goto_parts(goto_stmt);
        let label_id = self.tree.label_id(label);
        let nested = self.tree.sibling_from_nephew(goto_stmt, label);
        let variable = self.tree.expr(Expr::Variable(label_id));
        let loop_stmt = self.tree.create(Kind::Loop { cond: variable }, parent);
        self.tree.splice(loop_stmt, parent, nested, goto_stmt);
        let mut cur = self.tree.s(loop_stmt).first;
        while cur != NIL {
            if matches!(self.tree.s(cur).kind, Kind::Break { .. }) {
                return Err(TranslateError::Unstructured { address: 0 });
            }
            cur = self.tree.s(cur).next;
        }
        self.tree.insert(parent, goto_stmt, loop_stmt);
        let new_goto = self.tree.create(
            Kind::Goto {
                cond: variable,
                label,
            },
            loop_stmt,
        );
        self.tree.push_front(loop_stmt, new_goto);
        let set_var = self.tree.create(
            Kind::SetVariable {
                id: label_id,
                op: goto_cond,
            },
            loop_stmt,
        );
        self.tree.push_back(loop_stmt, set_var);
        self.tree.erase(goto_stmt);
        Ok(new_goto)
    }

    fn move_outward_if(&mut self, goto_stmt: StmtId) -> StmtId {
        let parent = self.tree.s(goto_stmt).up;
        let (goto_cond, label) = self.tree.goto_parts(goto_stmt);
        let label_id = self.tree.label_id(label);
        let set_var = self.tree.create(
            Kind::SetVariable {
                id: label_id,
                op: goto_cond,
            },
            parent,
        );
        self.tree.insert(parent, goto_stmt, set_var);
        let first = self.tree.s(goto_stmt).next;
        if first != NIL {
            let cond = self.tree.expr(Expr::Variable(label_id));
            let neg = self.tree.expr(Expr::Not(cond));
            let if_stmt = self.tree.create(Kind::If { cond: neg }, parent);
            self.tree.splice(if_stmt, parent, first, NIL);
            self.tree.insert(parent, goto_stmt, if_stmt);
        }
        self.tree.erase(goto_stmt);
        let new_cond = self.tree.expr(Expr::Variable(label_id));
        let grandparent = self.tree.s(parent).up;
        let new_goto = self.tree.create(
            Kind::Goto {
                cond: new_cond,
                label,
            },
            grandparent,
        );
        let after = self.tree.s(parent).next;
        self.tree.insert(grandparent, after, new_goto)
    }

    fn move_outward_loop(&mut self, goto_stmt: StmtId) -> StmtId {
        let parent = self.tree.s(goto_stmt).up;
        let (goto_cond, label) = self.tree.goto_parts(goto_stmt);
        let label_id = self.tree.label_id(label);
        let set_var = self.tree.create(
            Kind::SetVariable {
                id: label_id,
                op: goto_cond,
            },
            parent,
        );
        let cond = self.tree.expr(Expr::Variable(label_id));
        let break_stmt = self.tree.create(Kind::Break { cond }, parent);
        self.tree.insert(parent, goto_stmt, set_var);
        self.tree.insert(parent, goto_stmt, break_stmt);
        self.tree.erase(goto_stmt);
        let new_cond = self.tree.expr(Expr::Variable(label_id));
        let grandparent = self.tree.s(parent).up;
        let new_goto = self.tree.create(
            Kind::Goto {
                cond: new_cond,
                label,
            },
            grandparent,
        );
        let after = self.tree.s(parent).next;
        self.tree.insert(grandparent, after, new_goto)
    }
}

/// Translates guest code in `[begin, end)` (program offsets) into a block.
pub type TranslateFn<'a> =
    dyn FnMut(&mut Program, BlockId, u32, u32) -> Result<(), TranslateError> + 'a;

/// Options that change the structure of the output.
#[derive(Clone, Copy, Debug, Default)]
pub struct StructureOptions {
    /// Route every exit through one return at the end (task and mesh programs emit their outputs
    /// there).
    pub converge_exits: bool,
    /// Move conditional demotes to just before the single return (hosts whose demote does not
    /// keep helper invocations running).
    pub reorder_demotes: bool,
}

fn visit_expr(exprs: &[Expr], ir: &mut Emitter<'_>, expr: ExprId) -> Value {
    match exprs[expr as usize] {
        Expr::Identity(cond) => ir.condition(cond),
        Expr::Not(op) => {
            let v = visit_expr(exprs, ir, op);
            ir.logical_not(v)
        }
        Expr::Or(a, b) => {
            let a = visit_expr(exprs, ir, a);
            let b = visit_expr(exprs, ir, b);
            ir.logical_or(a, b)
        }
        Expr::Variable(id) => ir.get_goto_variable(id),
        Expr::IndirectBranchCond(location) => {
            let v = ir.get_indirect_branch_variable();
            ir.icompare(ICompare::Equal, v, Value::U32(location))
        }
    }
}

struct TranslatePass<'a, 'b> {
    tree: &'a mut Tree,
    prog: &'a mut Program,
    translate: &'a mut TranslateFn<'b>,
    uses_demote: bool,
}

impl TranslatePass<'_, '_> {
    fn new_block(&mut self) -> BlockId {
        self.prog.new_block()
    }

    fn cond_ref(&mut self, block: BlockId, expr: ExprId) -> Value {
        let mut ir = Emitter::new(self.prog, block);
        let v = visit_expr(&self.tree.exprs, &mut ir, expr);
        ir.condition_ref(v)
    }

    /// The block that follows `stmt`; a placeholder code statement keeps the following code
    /// in the same block.
    fn merge_block(&mut self, parent: StmtId, stmt: StmtId) -> BlockId {
        if self.tree.try_find_forward_block(stmt).is_none() {
            let dummy = self.tree.create(Kind::Code { begin: 0, end: 0 }, parent);
            let after = self.tree.s(stmt).next;
            self.tree.insert(parent, after, dummy);
        }
        self.new_block()
    }

    fn visit(
        &mut self,
        parent: StmtId,
        break_block: Option<BlockId>,
        fallthrough: Option<BlockId>,
    ) -> Result<(), TranslateError> {
        let mut current: Option<BlockId> = None;
        let mut stmt = self.tree.s(parent).first;
        while stmt != NIL {
            let kind = self.tree.s(stmt).kind;
            match kind {
                Kind::Label { .. } => {}
                Kind::Code { begin, end } => {
                    let block = self.ensure_block(&mut current);
                    if begin != end {
                        (self.translate)(self.prog, block, begin, end)?;
                    }
                }
                Kind::SetVariable { id, op } => {
                    let block = self.ensure_block(&mut current);
                    let mut ir = Emitter::new(self.prog, block);
                    let v = visit_expr(&self.tree.exprs, &mut ir, op);
                    ir.set_goto_variable(id, v);
                }
                Kind::SetIndirectBranchVariable { reg, offset } => {
                    let block = self.ensure_block(&mut current);
                    let mut ir = Emitter::new(self.prog, block);
                    let r = ir.get_reg(reg);
                    let address = ir.iadd(r, Value::U32(offset as u32));
                    ir.set_indirect_branch_variable(address);
                }
                Kind::If { cond } => {
                    let block = self.ensure_block(&mut current);
                    let merge = self.merge_block(parent, stmt);
                    let cond = self.cond_ref(block, cond);
                    let if_index = self.prog.syntax_list.len();
                    self.prog.syntax_list.push(SyntaxNode::Unreachable);
                    let then_index = self.prog.syntax_list.len();
                    self.visit(stmt, break_block, Some(merge))?;
                    let then_block = self.syntax_block(then_index)?;
                    self.prog.add_branch(block, then_block);
                    self.prog.add_branch(block, merge);
                    current = Some(merge);
                    self.prog.syntax_list[if_index] = SyntaxNode::If {
                        cond,
                        body: then_block,
                        merge,
                    };
                    self.prog.syntax_list.push(SyntaxNode::EndIf { merge });
                    self.prog.syntax_list.push(SyntaxNode::Block(merge));
                }
                Kind::Loop { cond } => {
                    if self.prog.syntax_list.is_empty() {
                        self.ensure_block(&mut current);
                    }
                    let header = self.new_block();
                    if let Some(block) = current {
                        self.prog.add_branch(block, header);
                    }
                    self.prog.syntax_list.push(SyntaxNode::Block(header));
                    let continue_block = self.new_block();
                    let merge = self.merge_block(parent, stmt);
                    let loop_index = self.prog.syntax_list.len();
                    self.prog.syntax_list.push(SyntaxNode::Unreachable);
                    let body_index = self.prog.syntax_list.len();
                    self.visit(stmt, Some(merge), Some(continue_block))?;
                    let cond = self.cond_ref(continue_block, cond);
                    let body = self.syntax_block(body_index)?;
                    self.prog.add_branch(header, body);
                    self.prog.add_branch(continue_block, header);
                    self.prog.add_branch(continue_block, merge);
                    current = Some(merge);
                    self.prog.syntax_list[loop_index] = SyntaxNode::Loop {
                        body,
                        continue_block,
                        merge,
                    };
                    self.prog
                        .syntax_list
                        .push(SyntaxNode::Block(continue_block));
                    self.prog.syntax_list.push(SyntaxNode::Repeat {
                        cond,
                        loop_header: header,
                        merge,
                    });
                    self.prog.syntax_list.push(SyntaxNode::Block(merge));
                }
                Kind::Break { cond } => {
                    let block = self.ensure_block(&mut current);
                    let skip = self.merge_block(parent, stmt);
                    let cond = self.cond_ref(block, cond);
                    let target = break_block.ok_or(TranslateError::Unstructured { address: 0 })?;
                    self.prog.add_branch(block, target);
                    self.prog.add_branch(block, skip);
                    current = Some(skip);
                    self.prog.syntax_list.push(SyntaxNode::Break {
                        cond,
                        merge: target,
                        skip,
                    });
                    self.prog.syntax_list.push(SyntaxNode::Block(skip));
                }
                Kind::Return => {
                    let block = self.ensure_block(&mut current);
                    let ret = self.new_block();
                    Emitter::new(self.prog, ret).epilogue();
                    self.prog.add_branch(block, ret);
                    self.prog.syntax_list.push(SyntaxNode::Block(ret));
                    self.prog.syntax_list.push(SyntaxNode::Return);
                    current = None;
                }
                Kind::Kill => {
                    let block = self.ensure_block(&mut current);
                    let demote = self.merge_block(parent, stmt);
                    Emitter::new(self.prog, block).demote_to_helper_invocation();
                    self.prog.add_branch(block, demote);
                    self.prog.syntax_list.push(SyntaxNode::Block(demote));
                    current = Some(demote);
                    self.uses_demote = true;
                }
                Kind::Unreachable => {
                    self.ensure_block(&mut current);
                    current = None;
                    self.prog.syntax_list.push(SyntaxNode::Unreachable);
                }
                Kind::Goto { .. } | Kind::Function => {
                    return Err(TranslateError::Unstructured { address: 0 });
                }
            }
            stmt = self.tree.s(stmt).next;
        }
        if let Some(block) = current {
            match fallthrough {
                Some(f) => self.prog.add_branch(block, f),
                None => self.prog.syntax_list.push(SyntaxNode::Unreachable),
            }
        }
        Ok(())
    }

    fn ensure_block(&mut self, current: &mut Option<BlockId>) -> BlockId {
        if let Some(b) = *current {
            return b;
        }
        let b = self.new_block();
        self.prog.syntax_list.push(SyntaxNode::Block(b));
        *current = Some(b);
        b
    }

    fn syntax_block(&self, index: usize) -> Result<BlockId, TranslateError> {
        match self.prog.syntax_list.get(index) {
            Some(&SyntaxNode::Block(b)) => Ok(b),
            _ => Err(TranslateError::Unstructured { address: 0 }),
        }
    }

    fn block_has(&self, block: BlockId, op: Opcode) -> bool {
        let mut cur = self.prog.block(block).first();
        while let Some(id) = cur {
            if self.prog.op(id) == op {
                return true;
            }
            cur = self.prog.inst(id).next();
        }
        false
    }

    /// Moves every top-level conditional demote outside loops to one conditional demote before
    /// the return. A demote inside a loop stays put: its condition holds per iteration.
    fn demote_combination(&mut self) {
        let list = &self.prog.syntax_list;
        let mut demote_blocks = Vec::new();
        let mut epilogues = 0;
        let mut depth = 0u32;
        let mut loop_depth = 0u32;
        for node in list {
            match *node {
                SyntaxNode::If { .. } => depth += 1,
                SyntaxNode::EndIf { .. } => depth = depth.saturating_sub(1),
                SyntaxNode::Loop { .. } => loop_depth += 1,
                SyntaxNode::Repeat { .. } => loop_depth = loop_depth.saturating_sub(1),
                SyntaxNode::Block(b) => {
                    if self.block_has(b, Opcode::DemoteToHelperInvocation) {
                        if depth <= 1 && loop_depth == 0 {
                            demote_blocks.push(b);
                        }
                    } else if self.block_has(b, Opcode::Epilogue) {
                        epilogues += 1;
                    }
                }
                _ => {}
            }
        }
        if demote_blocks.is_empty() || epilogues > 1 {
            return;
        }
        let mut conds = Vec::new();
        let mut start = 0;
        for &demote in &demote_blocks {
            let list = &mut self.prog.syntax_list;
            let Some(if_pos) = list[start..]
                .iter()
                .position(|n| matches!(n, SyntaxNode::If { body, .. } if *body == demote))
                .map(|p| p + start)
            else {
                return;
            };
            let SyntaxNode::If {
                cond,
                merge: end_if,
                ..
            } = list[if_pos]
            else {
                return;
            };
            conds.push(cond);
            start = if_pos;
            list.remove(if_pos);
            if let Some(p) = list[if_pos..]
                .iter()
                .position(|n| *n == SyntaxNode::Block(demote))
            {
                list.remove(if_pos + p);
            }
            if let Some(p) = list[if_pos..]
                .iter()
                .position(|n| *n == SyntaxNode::EndIf { merge: end_if })
            {
                list.remove(if_pos + p);
            }
        }
        let Some(ret_pos) = self.prog.syntax_list.iter().rposition(
            |n| matches!(n, SyntaxNode::Block(b) if self.block_has(*b, Opcode::Epilogue)),
        ) else {
            return;
        };
        let SyntaxNode::Block(return_block) = self.prog.syntax_list[ret_pos] else {
            return;
        };
        let Some(&SyntaxNode::Block(before)) = ret_pos
            .checked_sub(1)
            .and_then(|p| self.prog.syntax_list.get(p))
        else {
            return;
        };
        let mut ir = Emitter::new(self.prog, before);
        let mut cond = Value::U1(false);
        for c in conds {
            cond = ir.logical_or(cond, c);
        }
        let cond = ir.condition_ref(cond);
        let list = &mut self.prog.syntax_list;
        list.insert(
            ret_pos,
            SyntaxNode::EndIf {
                merge: return_block,
            },
        );
        list.insert(ret_pos, SyntaxNode::Block(demote_blocks[0]));
        list.insert(
            ret_pos,
            SyntaxNode::If {
                cond,
                body: demote_blocks[0],
                merge: return_block,
            },
        );
    }
}

/// Builds structured IR for `cfg` into `prog` (its blocks, edges and syntax list), calling
/// `translate` for every guest block in program order, then emits the prologue.
pub fn structure(
    prog: &mut Program,
    cfg: &Cfg,
    options: StructureOptions,
    translate: &mut TranslateFn<'_>,
) -> Result<(), TranslateError> {
    let mut tree = GotoPass::run(cfg, options.converge_exits)?;
    let mut pass = TranslatePass {
        tree: &mut tree,
        prog,
        translate,
        uses_demote: false,
    };
    pass.visit(ROOT, None, None)?;
    let Some(&SyntaxNode::Block(first)) = pass.prog.syntax_list.first() else {
        return Err(TranslateError::Unstructured { address: 0 });
    };
    pass.prog.prepend(first, Opcode::Prologue, &[], 0);
    if pass.uses_demote && options.reorder_demotes {
        pass.demote_combination();
    }
    pass.prog.block_order = pass
        .prog
        .syntax_list
        .iter()
        .filter_map(|n| match *n {
            SyntaxNode::Block(b) => Some(b),
            _ => None,
        })
        .collect();
    Ok(())
}
