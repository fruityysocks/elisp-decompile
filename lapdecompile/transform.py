import re
from copy import copy
from spark_parser import (
    GenericASTTraversal,
    GenericASTBuilder,
    GenericASTTraversalPruningException,
)
from lapdecompile.tok import Token
from lapdecompile.treenode import SyntaxTree

TRANSFORM = {}
def emacs_key_translate(s):
    result = ""
    if s[0] == '"':
        result = '"'
        for c in s[1:]:
            if ord(c) < 31:
                result += "\C-%s" % chr(ord("a") + ord(c) - 1)
            else:
                result += c
                pass
            pass
    else:
        m = re.match("^\[(\w+(?: \w+)*)\]$", s)
        if m:
            for s in m.group(1).split():
                try:
                    i = int(s)
                    if i == 27:
                        result += " ESC"
                    elif 134217728 <= i <= 134217759:
                        result += " C-M-%s" % chr(95 - (134217759 - i)).lower()
                    elif 134217761 <= i <= 134217854:
                        result += " M-%s" % chr(126 - (134217854 - i)).lower()
                except ValueError:
                    # FIXME: check that is something like "right" "up", etc.
                    # Also handle <C-M-down>
                    result += " <%s>" % s

    if result != s:
        if result.startswith('"'):
            return "(kbd %s)" % result.lstrip(" ")
        else:
            if result.endswith("\\"):
                result += "\\"
            return '(kbd "%s")' % result.lstrip(" ")
    return s


def emacs_key_normalize(name_expr_node):
    const_node = name_expr_node[0]
    s = const_node.attr
    result = emacs_key_translate(s)
    if result != s:
        const_node.kind = "TSTRING"
        const_node.attr = result.lstrip(" ")


class TransformTree(GenericASTTraversal, object):
    def __init__(self, ast, debug=False):
        self.debug = debug
        GenericASTTraversal.__init__(self, ast=None)
        return

    @staticmethod
    def unop_operator(node):
        return node[0].kind

    @staticmethod
    def binop_operator(node):
        return node[2][0].kind

    def preorder(self, node=None):
        """Walk the tree in roughly 'preorder' (a bit of a lie explained below).
        For each node with typestring name *name* if the
        node has a method called n_*name*, call that before walking
        children.

        In typical use a node with children can call "preorder" in any
        order it wants which may skip children or order then in ways
        other than first to last.  In fact, this this happens.  So in
        this sense this function not strictly preorder.
        """
        if node is None:
            node = self.ast

        try:
            name = "n_" + self.typestring(node)
            if hasattr(self, name):
                func = getattr(self, name)
                node = func(node)
        except GenericASTTraversalPruningException:
            return

        for i, kid in enumerate(node):
            node[i] = self.preorder(kid)
        return node

    def default(self, node):
        if not hasattr(node, "__len__"):
            return
        l = len(node)
        key = (node.kind, l)

        if key in TRANSFORM:
            entry = TRANSFORM[key]
            name = "n_%s_%d_%s_%d" % (node.kind, l, entry[0], entry[1])
            if hasattr(self, name):
                func = getattr(self, name)
                func(node)
            self.prune()

    def n_unary_expr_neg_eql(self, node):
        binary_expr = node[0][0]
        if not (
            node[0] == "expr" and binary_expr in ("binary_expr", "binary_expr_stacked")
        ):
            return node
        binary_op = self.binop_operator(binary_expr)
        unary_op = self.unop_operator(node[1])
        if unary_op == "NOT" and binary_op == "EQLSIGN":
            binary_expr[2][0].kind = "NEQLSIGN"
            node = SyntaxTree(
                node[0][0].kind, binary_expr, transformed_by="n_" + node.kind
            )
        return node

    def n_binary_expr(self, node):
        # Flatten f(a (f b))
        fn_name = node[2][0].kind
        if fn_name not in ("MIN", "MAX", "NCONC"):
            return node
        first_expr = node[0]
        if not (first_expr and first_expr[0] == "binary_expr"):
            return node

        arg_list = node[1]
        first_expr = first_expr[0]
        fn_name2 = first_expr[2][0].kind
        while fn_name == fn_name2:
            arg_list.append(first_expr[1])
            first_expr = first_expr[0]
            if first_expr != "expr":
                break
            first_expr = first_expr[0]
            if first_expr != "binary_expr":
                break
            fn_name2 = first_expr[2][0].kind

        arg_list.append(first_expr)
        arg_list.reverse()
        node = SyntaxTree(
            fn_name.lower() + "_exprn", arg_list, transformed_by="n_binary_expr"
        )
        return node

    def n_unary_expr(self, node):
        if node[0] == "expr":
            expr = node[0]
            unary_op1 = self.unop_operator(node[1])
            if expr[0] == "unary_expr":
                unary_op2 = self.unop_operator(expr[0][1])
                # Handle (cxr (xyr ... )) -> (cxyr ...)
                # FIXME: We combine only two functions. subr.el has up to 4 levels
                if re.match("C[AD]R", unary_op1) and re.match("C[AD]R", unary_op2):
                    c12r = f"C%s%sR" % (unary_op1[1:2], unary_op2[1:2])
                    expr[0][1][0].kind = c12r
                    node = SyntaxTree(
                        node.kind, expr[0], transformed_by="n_" + node.kind
                    )
                pass
        return node

    def n_call_exprn(self, call_expr):
        expr = call_expr[0]
        assert expr == "expr"
        if len(call_expr) == 4:
            call_expr = self.call_exprn_4_name_expr(call_expr, expr)
        elif expr[0] == "name_expr":
            name_expr = expr[0]
            if len(call_expr) == 3 and name_expr[0].attr.startswith(
                "(lambda (def-tmp-var) ("
            ):
                match = re.match(
                    '\(lambda \(def-tmp-var\) \((defvar|defconst) (.+) def-tmp-var( (".*"))?\)\)',
                    name_expr[0].attr,
                )
                if match:
                    def_type = match.group(1)
                    if match.group(4):
                        call_expr = SyntaxTree(
                            f"{def_type}_doc",
                            [
                                Token("CONSTANT", attr=match.group(2)),
                                call_expr[1],
                                Token("CONSTANT", attr=match.group(4)),
                            ],
                            transformed_by="n_call_exprn",
                        )
                    else:
                        call_expr = SyntaxTree(
                            f"{def_type}",
                            [Token("CONSTANT", attr=match.group(2)), call_expr[1]],
                            transformed_by="n_call_exprn",
                        )
            elif len(call_expr) == 5:
                call_expr = self.call_exprn_5_name_expr(call_expr, name_expr)
                pass
            pass
        return call_expr

    def call_exprn_4_name_expr(self, call_expr, expr):
        if expr[0] == "name_expr":
            fn_name = expr[0][0]
        else:
            assert expr[0] == "VARREF"
            fn_name = expr[0]

        if fn_name == "CONSTANT" and fn_name.attr in frozenset(
            ["global-set-key", "local-set-key"]
        ):
            key_expr = call_expr[1][0]
            if key_expr == "name_expr" and key_expr[0] == "CONSTANT":
                emacs_key_normalize(key_expr)
                pass
            pass
        return call_expr

    def call_exprn_5_name_expr(self, call_expr, name_expr):
        fn_name = name_expr[0]
        if fn_name == "CONSTANT" and fn_name.attr == "define-key":
            key_expr = call_expr[2][0]
            if key_expr == "name_expr" and key_expr[0] == "CONSTANT":
                emacs_key_normalize(key_expr)
                pass
            pass
        return call_expr

    def n_clause(self, node):
        body = node[1]
        if body != "body":
            return node
        exprs = body[0]
        assert exprs == "exprs"
        first_expr = exprs[0]
        if first_expr == "expr_stmt" and first_expr[0][0] == "if_form":
            # Remove if_form. That is
            # cond ((if (expr) ... )) =>
            # cond ((expr) ...)
            end_clause = node[-1]
            assert end_clause == "end_clause"
            if_form = first_expr[0][0]
            condition = SyntaxTree(
                "condition",
                [
                    if_form[0],
                    if_form[1],
                    SyntaxTree("opt_come_from", []),
                    SyntaxTree("opt_label", []),
                ],
            )
            inner_expr = SyntaxTree("expr", [if_form[2]], transformed_by="n_" + node.kind)
            exprs_node = SyntaxTree("exprs", [inner_expr], transformed_by="n_" + node.kind)
            body_node = SyntaxTree("body", [exprs_node], transformed_by="n_" + node.kind)

            node = SyntaxTree(
                "clause", [condition, body_node, end_clause], transformed_by="n_" + node.kind
            )
        return node

    def n_expr_stmt(self, node):
        expr = node[0]
        assert expr == "expr"
        expr_first = expr[0]
        if expr_first == "and_form" and len(expr_first) == 5:
            # Expect and_form structure like: ["and_form", cond, GOTO-IF-NIL, cond2, ...]
            # Use explicit indexing to gather the original child objects
            # (depends on how your parser stores `.data` — adapt as needed)

            # If the parser stores parts as items 1..n:
            parts = list(expr_first)  # shallow copy of children
            # Build a proper if_form with children as a list
            if_form = SyntaxTree("if_form", parts, transformed_by="n_" + node.kind)

            # Wrap if_form into expr, then expr_stmt, ensuring lists for children
            expr_wrapped = SyntaxTree("expr", [if_form], transformed_by="n_" + node.kind)
            node = SyntaxTree("expr_stmt", [expr_wrapped], transformed_by="n_" + node.kind)
        return node


    def n_let_form_star(self, let_form_star):
        assert len(let_form_star) >= 2
        varlist, body = let_form_star[:2]
        assert varlist == "varlist"
        assert body == "body"
        varbind = varlist[0]
        body_exprs = body[0]
        if (
            varbind == "varbind"
            and body_exprs == "exprs"
            and varbind[1].attr == "temp-buffer"
        ):
            if body_exprs[0][0][0] == "with_current_buffer_macro":
                with_current_buffer = body_exprs[0][0][0]
                body = body_exprs[1:]
                if len(body_exprs) == 1 and with_current_buffer[4] == "exprs":
                    unwind_protect_form = with_current_buffer[4][0][0][0]
                    if unwind_protect_form == "unwind_protect_form":
                        body = unwind_protect_form[2]
                        assert body == "opt_exprs"
                    pass
                # transform into "with-temp-buffer"
                body_exprs = SyntaxTree("exprs", body, transformed_by="n_let_form_star")
                with_temp_buffer_macro = SyntaxTree(
                    "with_temp_buffer_macro",
                    [body_exprs],
                    transformed_by="n_let_form_star",
                )
                return with_temp_buffer_macro
            pass
        return let_form_star

    def n_save_current_buffer_form(self, node):
        body = node[1]
        assert body == "body"
        exprs = body[0]
        assert exprs == "exprs"
        first_expr_stmt = exprs[0]
        assert first_expr_stmt == "expr_stmt"
        first_expr = first_expr_stmt[0]
        assert first_expr == "expr"
        set_buffer = first_expr[0]
        if set_buffer == "set_buffer":
            name_expr = set_buffer[0][0]
            if name_expr == "name_expr" and name_expr[0] == "VARREF":
                # Turn (save-buffer (set-buffer ...) (...)) into:
                # (with-current-buffer ...)
                exprs = SyntaxTree("exprs", exprs[1:], transformed_by="n_" + node.kind)
                node = SyntaxTree(
                    "with_current_buffer_macro",
                    [set_buffer[0][0], exprs],
                    transformed_by="n_" + node.kind,
                )
                pass
        return node

    def n_ternary_expr(self, node):
        expr2 = node[2]
        assert expr2 == "expr"
        expr2_first = expr2[0]
        ternary_op = node[3]
        assert ternary_op == "ternary_op"

        if (
            ternary_op[0] == "SUBSTRING"
            and expr2_first == "name_expr"
            and expr2_first[0].attr == "nil"
        ):
            binary_op = SyntaxTree(
                "binary_op", [ternary_op[0]], transformed_by="n_" + node.kind
            )
            node = SyntaxTree(
                "binary_expr",
                [node[0], node[1], binary_op],
                transformed_by="n_" + node.kind,
            )
        return node

    def n_when_macro(self, node):
        body = node[2]
        assert body == "body"
        if len(body[0]) == 1:
            node = SyntaxTree(
                "if_form", [node[0], node[1], node[2]], transformed_by="n_" + node.kind
            )
        return node

    def n_if_form(self, node):
        # Check if this is an if_form with a cond_form that should be an if-else
        # Pattern: if_form with 5 children where child[2] is a cond_form with 2 clauses
        # both starting with (t ...)
        if len(node) == 5 and node[2] == "expr":
            then_expr = node[2][0]
            if then_expr == "cond_form" and len(then_expr) == 2:
                # Check if it's a clause + labeled_clauses structure
                first_clause = then_expr[0]
                labeled_clauses = then_expr[1]

                # Check if both clauses are (t ...) style (unconditional)
                # This indicates it's really an if-else, not a cond
                if (first_clause == "clause" and
                    labeled_clauses == "labeled_clauses"):
                    # Extract the bodies from both clauses
                    # First clause structure: clause -> [body, end_clause]
                    # Labeled clause structure: labeled_clauses -> labeled_clause -> [LABEL, clause]
                    if (len(first_clause) >= 2 and
                        first_clause[0] == "body"):
                        # This looks like an if-else pattern
                        # Get the second clause from labeled_clauses
                        if labeled_clauses[0] == "labeled_clause":
                            labeled_clause = labeled_clauses[0]
                            if len(labeled_clause) >= 2 and labeled_clause[1] == "clause":
                                second_clause = labeled_clause[1]
                                if len(second_clause) >= 2 and second_clause[0] == "body":
                                    # Transform to if_else_form
                                    # if_else_form structure from semantic_consts.py line 54:
                                    # if_else_form ::= expr GOTO-IF-NIL expr GOTO come_froms LABEL expr opt_come_from opt_label
                                    # We need to construct: condition, goto-if-nil, then-expr, ..., else-expr
                                    # But our current structure is simpler, so let's create a custom structure
                                    # that the semantics can handle

                                    # For now, just extract the body expressions
                                    then_body = first_clause[0]  # body node
                                    else_body = second_clause[0]  # body node

                                    # Create an if_else_form-like structure
                                    # We'll use indices that match the semantic_consts.py template:
                                    # "if_else_form": ( "%(if %c\n%+%+%|%c\n%_%|%c)%_", 0, 2, 6 )
                                    # So we need: [condition, ..., then-expr, ..., ..., ..., else-expr]
                                    node = SyntaxTree(
                                        "if_else_form",
                                        [
                                            node[0],  # condition (index 0)
                                            node[1],  # GOTO-IF-NIL (index 1)
                                            then_body,  # then body (index 2)
                                            SyntaxTree("opt_come_from", []),  # index 3
                                            SyntaxTree("opt_label", []),  # index 4
                                            SyntaxTree("opt_come_from", []),  # index 5
                                            else_body,  # else body (index 6)
                                        ],
                                        transformed_by="n_" + node.kind,
                                    )
        return node

    def traverse(self, node):
        self.preorder(node)

    def transform(self, ast):
        # self.maybe_show_tree(ast)
        self.ast = copy(ast)
        self.ast = self.traverse(self.ast, is_lambda=False)
        return self.ast


if __name__ == "__main__":
    for seq in """
            [134217761]
            [134217854]
            [134217728]
            [134217729]
            [134217730]
            [134217759]
            [134217820]
            [134217822]
            [134217843|27]
            [27|right]
            """.split():
        print(
            "'%s' -> '%s'" % (seq, emacs_key_translate(seq.strip().replace("|", " ")))
        )
