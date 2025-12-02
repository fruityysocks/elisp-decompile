# lapdecompile/structure.py
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Optional, Set, Dict

from lapdecompile.treenode import SyntaxTree
from lapdecompile.tok import Token


def parse_offset(offset_str):
    if ':' in str(offset_str):
        # Format is "offset:label", extract just the offset
        return int(str(offset_str).split(':')[0])
    return int(offset_str)


def bytecode_to_ast(tokens):
    """Simulate stack machine to build AST from bytecode."""
    stack = []
    statements = []

    for token in tokens:
        kind = token.kind

        if kind == 'CONSTANT':
            stack.append(SyntaxTree("expr", [
                SyntaxTree("name_expr", [token])
            ]))

        elif kind == 'VARREF':
            stack.append(SyntaxTree("expr", [
                SyntaxTree("name_expr", [token])
            ]))

        # Grammar: setq_form ::= expr VARSET
        elif kind == 'VARSET':
            if stack:
                value = stack.pop()
                if value.kind != "expr":
                    value = SyntaxTree("expr", [value])

                setq = SyntaxTree("setq_form", [
                    value,    # index 0: expr (the value being assigned)
                    token     # index 1: VARSET token (contains variable name)
                ])
                stack.append(SyntaxTree("expr", [setq]))
            else:
                # Empty stack - treat as simple reference
                stack.append(SyntaxTree("expr", [SyntaxTree("name_expr", [token])]))

        # Binary operations
        elif kind in ('EQ', 'GTR', 'LSS', 'LEQ', 'GEQ', 'PLUS', 'DIFF', 'MULT', 'QUO', 'REM',
                     'MIN', 'MAX', 'NCONC'):
            if len(stack) >= 2:
                arg2 = stack.pop()
                arg1 = stack.pop()

                # Keep expr nodes as-is (binary_expr expects: expr expr binary_op)
                # Grammar: binary_expr ::= expr expr binary_op
                if arg1.kind != "expr":
                    arg1 = SyntaxTree("expr", [arg1])
                if arg2.kind != "expr":
                    arg2 = SyntaxTree("expr", [arg2])

                binary = SyntaxTree("binary_expr", [
                    arg1,              # index 0: expr
                    arg2,              # index 1: expr
                    SyntaxTree("binary_op", [token])  # index 2: binary_op
                ])
                stack.append(SyntaxTree("expr", [binary]))
            else:
                stack.append(SyntaxTree("expr", [SyntaxTree("name_expr", [token])]))

        # Function calls with varying argument counts
        elif kind == 'CALL_0':
            if stack:
                func = stack.pop()

                call = SyntaxTree("call_expr0", [func, token])
                stack.append(SyntaxTree("expr", [call]))
            else:
                stack.append(SyntaxTree("expr", [SyntaxTree("name_expr", [token])]))

        elif kind == 'CALL_1':
            if len(stack) >= 2:
                arg = stack.pop()
                func = stack.pop()

                call = SyntaxTree("call_expr1", [func, arg, token])
                stack.append(SyntaxTree("expr", [call]))
            else:
                stack.append(SyntaxTree("expr", [SyntaxTree("name_expr", [token])]))

        elif kind == 'CALL_2':
            if len(stack) >= 3:
                arg2 = stack.pop()
                arg1 = stack.pop()
                func = stack.pop()

                call = SyntaxTree("call_expr2", [func, arg1, arg2, token])
                stack.append(SyntaxTree("expr", [call]))
            else:
                stack.append(SyntaxTree("expr", [SyntaxTree("name_expr", [token])]))

        elif kind == 'CALL_3':
            if len(stack) >= 4:
                arg3 = stack.pop()
                arg2 = stack.pop()
                arg1 = stack.pop()
                func = stack.pop()

                call = SyntaxTree("ternary_expr", [func, arg1, arg2, arg3])
                stack.append(SyntaxTree("expr", [call]))
            else:
                stack.append(SyntaxTree("expr", [SyntaxTree("name_expr", [token])]))

        elif kind == 'DUP':
            # In the AST, we treat it as an expression placeholder
            stack.append(SyntaxTree("expr", [
                SyntaxTree("name_expr", [token])
            ]))

        elif kind == 'DISCARD':
            if stack:
                expr = stack.pop()
                expr_stmt = SyntaxTree("expr_stmt", [expr, SyntaxTree("opt_discard", [token])])
                statements.append(expr_stmt)

        # List operations
        elif kind == 'CONS':
            if len(stack) >= 2:
                cdr = stack.pop()
                car = stack.pop()

                if car.kind == "expr" and len(car) > 0:
                    car = car[0]
                if cdr.kind == "expr" and len(cdr) > 0:
                    cdr = cdr[0]

                cons_expr = SyntaxTree("binary_expr", [
                    SyntaxTree("name_expr", [Token('SYMBOL', 'cons', token.offset)]),
                    car,
                    cdr
                ])
                stack.append(SyntaxTree("expr", [cons_expr]))
            else:
                stack.append(SyntaxTree("expr", [SyntaxTree("name_expr", [token])]))

        # Unary operations
        elif kind in ('NOT', 'CAR', 'CDR', 'NEGATE', 'SUB1', 'ADD1'):
            if stack:
                arg = stack.pop()

                # Build unary expression: arg unary_op
                # Grammar: unary_expr ::= expr unary_op
                unary = SyntaxTree("unary_expr", [
                    arg if arg.kind == "expr" else SyntaxTree("expr", [arg]),
                    SyntaxTree("unary_op", [token])
                ])
                stack.append(SyntaxTree("expr", [unary]))
            else:
                stack.append(SyntaxTree("expr", [SyntaxTree("name_expr", [token])]))

        elif kind in ('GOTO', 'GOTO-IF-NIL', 'GOTO-IF-NOT-NIL',
                     'GOTO-IF-NIL-ELSE-POP', 'GOTO-IF-NOT-NIL-ELSE-POP',
                     'LABEL', 'RETURN', 'COME_FROM', 'VARBIND', 'UNBIND'):
            # Skip control flow and binding instructions
            continue

        # Unknown opcode
        else:
            stack.append(SyntaxTree("expr", [
                SyntaxTree("name_expr", [token])
            ]))

    exprs = []
    exprs.extend(statements)

    for item in stack:
        if item.kind == "expr":
            expr_stmt = SyntaxTree("expr_stmt", [
                item,
                SyntaxTree("opt_discard", [])
            ])
            exprs.append(expr_stmt)
        else:
            expr_stmt = SyntaxTree("expr_stmt", [
                SyntaxTree("expr", [item]),
                SyntaxTree("opt_discard", [])
            ])
            exprs.append(expr_stmt)

    return SyntaxTree("body", [
        SyntaxTree("exprs", exprs if exprs else [])
    ])


# Structured IR node types
@dataclass
class BlockNode:
    block: object
    stmts: List[object] = None

    kind: str = "block"

    def __post_init__(self):
        if self.stmts is None:
            self.stmts = []

    def to_ast(self, parser, all_tokens):
        block_tokens = [
            t for t in all_tokens
            if self.block.start_offset <= parse_offset(t.offset) <= self.block.end_offset
        ]

        if not block_tokens:
            return SyntaxTree("body", [SyntaxTree("exprs", [])])

        result = bytecode_to_ast(block_tokens)
        if result.kind == "body" and len(result) > 0:
            exprs = result[0]
            if exprs.kind == "exprs" and len(exprs) > 3:
                simple_count = 0
                for expr_stmt in exprs:
                    if expr_stmt.kind == "expr_stmt" and len(expr_stmt) > 0:
                        expr = expr_stmt[0]
                        if expr.kind == "expr" and len(expr) > 0:
                            child = expr[0]
                            if child.kind == "name_expr":
                                simple_count += 1

                if simple_count > len(exprs) * 0.6:
                    try:
                        p = parser(None, block_tokens)
                        p.add_custom_rules(block_tokens, {})
                        parsed = p.parse(block_tokens)
                        if parsed:
                            return parsed
                    except:
                        pass

        return result


@dataclass
class SeqNode:
    parts: List[object]
    kind: str = "seq"

    @property
    def items(self):
        return self.parts

    def to_ast(self, parser, all_tokens):
        expr_asts = []
        for part in self.parts:
            if part is None:
                continue
            part_ast = part.to_ast(parser, all_tokens)
            if part_ast.kind == "body" and len(part_ast) > 0:
                exprs_node = part_ast[0]
                if exprs_node.kind == "exprs":
                    for expr in exprs_node:
                        expr_asts.append(expr)
                else:
                    expr_asts.append(exprs_node)
            else:
                expr_asts.append(part_ast)

        return SyntaxTree("body", [
            SyntaxTree("exprs", expr_asts)
        ])


@dataclass
class IfNode:
    test_block: object
    then_node: object
    else_node: Optional[object]
    join_block: Optional[object]

    kind: str = "if"

    @property
    def cond(self):
        return BlockNode(self.test_block)

    @property
    def then(self):
        return self.then_node

    @property
    def else_branch(self):
        return self.else_node

    def to_ast(self, parser, all_tokens):
        
        test_tokens = [
            t for t in all_tokens
            if self.test_block.start_offset <= parse_offset(t.offset) <= self.test_block.end_offset
        ]

        goto_token = None
        cond_tokens = []
        for t in test_tokens:
            if 'GOTO-IF' in t.kind:
                goto_token = t
                break
            cond_tokens.append(t)

        if cond_tokens:
            body_ast = bytecode_to_ast(cond_tokens)

            # Extract the expression from body -> exprs -> expr_stmt -> expr
            if body_ast.kind == "body" and len(body_ast) > 0:
                exprs = body_ast[0]
                if exprs.kind == "exprs" and len(exprs) > 0:
                    expr_stmt = exprs[0]
                    if expr_stmt.kind == "expr_stmt" and len(expr_stmt) > 0:
                        cond_ast = expr_stmt[0]  
                    else:
                        cond_ast = expr_stmt
                else:
                    cond_ast = SyntaxTree("expr", [])
            else:
                cond_ast = SyntaxTree("expr", [])
        else:
            cond_ast = SyntaxTree("expr", [])

        then_ast = self.then_node.to_ast(parser, all_tokens) if self.then_node else SyntaxTree("body", [])
        else_ast = self.else_node.to_ast(parser, all_tokens) if self.else_node else None

        def is_empty_body(ast):
            if ast is None:
                return True
            if ast.kind == "body" and len(ast) > 0:
                exprs = ast[0]
                if exprs.kind == "exprs" and len(exprs) == 0:
                    return True
            return False

        if is_empty_body(else_ast):
            else_ast = None

        if else_ast:
            return SyntaxTree("if_else_form", [
                cond_ast,                           # index 0: condition
                goto_token or SyntaxTree("GOTO-IF-NIL", []),  # index 1: goto instruction
                then_ast,                           # index 2: then body
                SyntaxTree("opt_come_from", []),    # index 3
                SyntaxTree("opt_label", []),        # index 4
                SyntaxTree("opt_come_from", []),    # index 5
                else_ast,                           # index 6: else body
            ])
        else:
            # If-then form (no else)
            return SyntaxTree("if_form", [
                cond_ast,                           # index 0: condition
                goto_token or SyntaxTree("GOTO-IF-NIL", []),
                then_ast,                           # index 2: then body
            ])


@dataclass
class WhileNode:
    header_block: object
    body_node: object

    kind: str = "while"

    @property
    def cond(self):
        return BlockNode(self.header_block)

    @property
    def body(self):
        return self.body_node

    def to_ast(self, parser, all_tokens):
        
        header_tokens = [
            t for t in all_tokens
            if self.header_block.start_offset <= parse_offset(t.offset) <= self.header_block.end_offset
        ]

        goto_token = None
        cond_tokens = []
        for t in header_tokens:
            if 'GOTO-IF' in t.kind or t.kind == 'GOTO':
                goto_token = t
                break
            cond_tokens.append(t)

        if cond_tokens:
            body_ast = bytecode_to_ast(cond_tokens)

            # Extract the expression from body -> exprs -> expr_stmt -> expr
            if body_ast.kind == "body" and len(body_ast) > 0:
                exprs = body_ast[0]
                if exprs.kind == "exprs" and len(exprs) > 0:
                    expr_stmt = exprs[0]
                    if expr_stmt.kind == "expr_stmt" and len(expr_stmt) > 0:
                        cond_ast = expr_stmt[0]  
                    else:
                        cond_ast = expr_stmt
                else:
                    cond_ast = SyntaxTree("expr", [])
            else:
                cond_ast = SyntaxTree("expr", [])
        else:
            cond_ast = SyntaxTree("expr", [])

        if self.body_node:
            body_ast = self.body_node.to_ast(parser, all_tokens)
        else:
            body_ast = SyntaxTree("body", [SyntaxTree("exprs", [])])

        return SyntaxTree("while_form2", [
            SyntaxTree("opt_label", []),        # index 0
            SyntaxTree("opt_come_from", []),    # index 1
            cond_ast,                           # index 2 - condition
            SyntaxTree("opt_label", []),        # index 3
            body_ast,                           # index 4 - body
        ])


@dataclass
class CondNode:
    clauses: List[dict]  
    kind: str = "cond"

    def to_ast(self, parser, all_tokens):
        clause_asts = []

        for i, clause_info in enumerate(self.clauses):
            test_block = clause_info.get('test_block')
            then_block = clause_info.get('then_block')

            if test_block is None:
                # Default clause (t ...)
                cond_ast = SyntaxTree("expr", [
                    SyntaxTree("name_expr", [Token('CONSTANT', 't', 0)])
                ])
            else:
                test_tokens = [
                    t for t in all_tokens
                    if test_block.start_offset <= parse_offset(t.offset) <= test_block.end_offset
                ]
                cond_tokens = [t for t in test_tokens if 'GOTO' not in t.kind]

                if cond_tokens:
                    body_ast = bytecode_to_ast(cond_tokens)

                # Extract the expression from body -> exprs -> expr_stmt -> expr
                    if body_ast.kind == "body" and len(body_ast) > 0:
                        exprs = body_ast[0]
                        if exprs.kind == "exprs" and len(exprs) > 0:
                            expr_stmt = exprs[0]
                            if expr_stmt.kind == "expr_stmt" and len(expr_stmt) > 0:
                                cond_ast = expr_stmt[0]  
                            else:
                                cond_ast = expr_stmt
                        else:
                            cond_ast = SyntaxTree("expr", [Token('CONSTANT', 't', 0)])
                    else:
                        cond_ast = SyntaxTree("expr", [Token('CONSTANT', 't', 0)])
                else:
                    cond_ast = SyntaxTree("expr", [Token('CONSTANT', 't', 0)])

            if clause_info.get('then_node'):
                then_ast = clause_info['then_node'].to_ast(parser, all_tokens)
            else:
                then_node = BlockNode(then_block)
                then_ast = then_node.to_ast(parser, all_tokens)

            # Check if body is empty (to prevent IndexError in transform.py)
            is_empty = False
            if then_ast.kind == "body" and len(then_ast) > 0:
                exprs = then_ast[0]
                if exprs.kind == "exprs" and len(exprs) == 0:
                    is_empty = True

            # If empty, create a placeholder nil expression
            if is_empty:
                then_ast = SyntaxTree("body", [
                    SyntaxTree("exprs", [
                        SyntaxTree("expr_stmt", [
                            SyntaxTree("expr", [
                                SyntaxTree("name_expr", [Token('CONSTANT', 'nil', 0)])
                            ]),
                            SyntaxTree("opt_discard", [])
                        ])
                    ])
                ])

            clause_ast = SyntaxTree("clause", [
                cond_ast,
                then_ast,
                SyntaxTree("end_clause", [Token('RETURN', '', 0)])
            ])

            clause_asts.append(clause_ast)

        return SyntaxTree("cond_form", [
            clause_asts[0] if clause_asts else SyntaxTree("clause", []),
            SyntaxTree("labeled_clauses", clause_asts[1:] if len(clause_asts) > 1 else [])
        ])


@dataclass
class DolistNode:
    list_expr: object      
    var_name: str          
    body_node: object      
    result_expr: Optional[object] = None  

    kind: str = "dolist"

    def to_ast(self, parser, all_tokens):
        if self.body_node:
            body_ast = self.body_node.to_ast(parser, all_tokens)
        else:
            body_ast = SyntaxTree("body", [SyntaxTree("exprs", [])])
        dolist_tokens = []
        found_list = False
        found_nil = False
        found_var = False
        found_dup = False
        found_tail = False

        for tok in all_tokens:
            if not found_list and tok.kind == 'CONSTANT':
                dolist_tokens.append(tok)
                found_list = True
            elif found_list and not found_nil and tok.kind == 'CONSTANT':
                nil_token = tok
                found_nil = True
            elif found_nil and not found_var and tok.kind == 'VARBIND' and tok.attr == self.var_name:
                var_token = tok
                found_var = True
            elif found_var and not found_dup and tok.kind == 'DUP':
                dup_token = tok
                found_dup = True
            elif found_dup and not found_tail and tok.kind == 'VARBIND' and tok.attr == '--dolist-tail--':
                tail_token = tok
                found_tail = True
                break

        if not (found_list and found_nil and found_var and found_dup and found_tail):
            # Fallback: build simplified structure
            var_token = Token('SYMBOL', self.var_name, 0)
            return SyntaxTree("dolist_macro", [
                self.list_expr,
                var_token,
                body_ast
            ])

        dolist_list = SyntaxTree("dolist_list", [self.list_expr])
        varbind = SyntaxTree("varbind", [
            SyntaxTree("expr", [SyntaxTree("name_expr", [nil_token])]),
            var_token
        ])
        dolist_init_var = SyntaxTree("dolist_init_var", [varbind, dup_token, tail_token])

        dummy_token = Token('GOTO-IF-NIL-ELSE-POP', '2', 0)

        return SyntaxTree("dolist_macro", [
            dolist_list,
            dolist_init_var,
            dummy_token,
            dummy_token,
            dummy_token,
            SyntaxTree("dolist_loop_iter_set", []),
            body_ast,
            dummy_token,
            dummy_token,
            dummy_token,
            dummy_token,
            dummy_token,
            dummy_token,
            dummy_token
        ])


@dataclass
class WhenNode:
    test_block: object
    then_node: object
    kind: str = "when"

    def to_ast(self, parser, all_tokens):
        
        test_tokens = [
            t for t in all_tokens
            if self.test_block.start_offset <= parse_offset(t.offset) <= self.test_block.end_offset
        ]

        cond_tokens = [t for t in test_tokens if 'GOTO' not in t.kind]

        if cond_tokens:
            body_ast = bytecode_to_ast(cond_tokens)

            # Extract the expression from body -> exprs -> expr_stmt -> expr
            if body_ast.kind == "body" and len(body_ast) > 0:
                exprs = body_ast[0]
                if exprs.kind == "exprs" and len(exprs) > 0:
                    expr_stmt = exprs[0]
                    if expr_stmt.kind == "expr_stmt" and len(expr_stmt) > 0:
                        cond_ast = expr_stmt[0]  
                    else:
                        cond_ast = expr_stmt
                else:
                    cond_ast = SyntaxTree("expr", [])
            else:
                cond_ast = SyntaxTree("expr", [])
        else:
            cond_ast = SyntaxTree("expr", [])

        then_ast = self.then_node.to_ast(parser, all_tokens) if self.then_node else SyntaxTree("body", [])

        if then_ast.kind != "body":
            then_ast = SyntaxTree("body", [
                SyntaxTree("exprs", [
                    SyntaxTree("expr_stmt", [
                        SyntaxTree("expr", [then_ast]),  
                        SyntaxTree("opt_discard", [])
                    ])
                ])
            ])

        return SyntaxTree("when_macro", [
            cond_ast,           
            SyntaxTree("GOTO-IF-NIL", []),
            then_ast,          
        ])

@dataclass
class ReturnNode:
    block: object
    kind: str = "return"

    @property
    def value(self):
        return BlockNode(self.block)



def block_by_offset(cfg, offset):
    return cfg.block_offsets.get(offset)

def find_common_successor(cfg, a, b):
    # BFS from a
    def reachable_from(src):
        seen = set()
        q = deque([src])
        while q:
            cur = q.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            for s in cur.successors:
                if s not in seen:
                    q.append(s)
        return seen

    ra = reachable_from(a)
    rb = reachable_from(b)
    inter = ra & rb
    if not inter:
        return None
    return min(inter, key=lambda bl: bl.start_offset)

def collect_region(entry, exit_block):
    seen = set()
    q = deque([entry])
    while q:
        cur = q.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        for s in cur.successors:
            if s == exit_block:
                continue
            if s not in seen:
                q.append(s)
    return seen

# Pattern recognizers --------------------------------------------------------

def is_conditional_header(block):
    try:
        tokens = getattr(block, "tokens", None) or getattr(block, "instrs", None)
        if tokens:
            for t in reversed(tokens):
                k = getattr(t, "kind", getattr(t, "opname", None)) or t
                if isinstance(k, str) and k.startswith("GOTO-IF-"):
                    return True
    except Exception:
        pass
    return (len(block.jump_offsets) >= 1) and (block.follow_offset is not None)

def is_simple_return_block(block):
    tokens = getattr(block, "tokens", None) or getattr(block, "instrs", None)
    if not tokens:
        return False
    for t in tokens:
        k = getattr(t, "kind", getattr(t, "opname", None)) or t
        if k == "RETURN":
            return True
    return False

def has_backward_edge(block):
    for j in getattr(block, "jump_offsets", ()):
        if j < block.start_offset:
            return True
    return False


def detect_dolist_pattern(cfg, entry_block, all_tokens):
    """Detect dolist macro bytecode pattern."""
    entry_tokens = [
        t for t in all_tokens
        if entry_block.start_offset <= parse_offset(t.offset) <= entry_block.end_offset
    ]

    if len(entry_tokens) < 5:
        return None

    dolist_var = None
    list_token = None
    has_dolist_tail = False
    nil_idx = None

    for i, tok in enumerate(entry_tokens):
        if tok.kind == 'CONSTANT' and tok.attr == 'nil':
            if (i + 3 < len(entry_tokens) and
                entry_tokens[i + 1].kind == 'VARBIND' and
                entry_tokens[i + 2].kind == 'DUP' and
                i + 3 < len(entry_tokens) and
                entry_tokens[i + 3].kind == 'VARBIND' and
                entry_tokens[i + 3].attr == '--dolist-tail--'):
                nil_idx = i
                dolist_var = entry_tokens[i + 1].attr
                has_dolist_tail = True
                break

    if nil_idx is None:
        return None

    for i in range(nil_idx - 1, -1, -1):
        tok = entry_tokens[i]
        if tok.kind not in ('LABEL', 'COME_FROM'):
            list_token = tok if tok.kind in ('CONSTANT', 'VARREF') else tok
            break

    if not (dolist_var and has_dolist_tail and list_token):
        return None

    has_conditional_exit = any(t.kind == 'GOTO-IF-NIL-ELSE-POP' for t in entry_tokens)
    if not has_conditional_exit or len(entry_block.successors) != 2:
        return None

    body_block = None
    exit_block = None

    for succ in entry_block.successors:
        has_loop = succ in succ.successors
        if not has_loop:
            visited = set()
            queue = deque([succ])
            while queue and not has_loop:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                if entry_block in current.successors or succ in current.successors:
                    has_loop = True
                    break
                for s in current.successors:
                    if s not in visited:
                        queue.append(s)

        if has_loop:
            body_block = succ
        else:
            exit_block = succ

    if not body_block:
        return None

    list_ast = SyntaxTree("expr", [SyntaxTree("name_expr", [list_token])])
    return (dolist_var, list_ast, entry_block, body_block, exit_block)


def detect_cond_pattern(cfg, entry_block, region):
    clauses = []
    current = entry_block
    merge_points = set()
    max_chain_length = 10  # Prevent infinite loops

    for _ in range(max_chain_length):
        if not is_conditional_header(current):
            break

        then_block = None
        else_block = None

        if getattr(current, "follow_offset", None):
            then_block = cfg.block_offsets.get(current.follow_offset)
        jump_offsets = getattr(current, "jump_offsets", []) or []
        jump_offsets_list = sorted(jump_offsets)
        else_block = cfg.block_offsets.get(jump_offsets_list[0]) if jump_offsets_list else None

        if not then_block or not else_block:
            break

        join = find_common_successor(cfg, then_block, else_block)
        if join:
            merge_points.add(join)

        clauses.append({
            'test_block': current,
            'then_block': then_block,
            'join': join
        })

        if is_conditional_header(else_block) and else_block in region:
            current = else_block
        else:
            clauses.append({
                'test_block': None,  
                'then_block': else_block,
                'join': join
            })
            break

    if len(clauses) > 2 and len(merge_points) <= 2:
        return (True, clauses)
    else:
        return (False, None)

def is_or_pattern(entry_block):
    if hasattr(entry_block, 'code') and entry_block.code:
        last_inst = entry_block.code[-1]
        if hasattr(last_inst, 'kind') and 'GOTO-IF-NOT-NIL' in last_inst.kind:
            return True

    return False


def is_when_pattern(then_node, else_node):
    if else_node is None:
        return False

    if isinstance(else_node, BlockNode):
        block = else_node.block
        tokens = getattr(block, "tokens", []) or []
        non_control_tokens = [
            t for t in tokens
            if t.kind not in ('LABEL', 'RETURN', 'COME_FROM', 'GOTO',
                            'GOTO-IF-NIL', 'GOTO-IF-NOT-NIL',
                            'GOTO-IF-NIL-ELSE-POP', 'GOTO-IF-NOT-NIL-ELSE-POP')
        ]
        return len(non_control_tokens) == 0

    return False


def structure_cfg(cfg, all_tokens):
    entry_block = cfg.entry_node
    visited = set()
    return _structure_region(cfg, entry_block, None, visited, all_tokens)

def _structure_region(cfg, entry_block, exit_block, visited, all_tokens):
    if entry_block in visited:
        return BlockNode(block=entry_block)

    region = collect_region(entry_block, exit_block)
    if not region:
        return None

    visited.add(entry_block)

    dolist_result = detect_dolist_pattern(cfg, entry_block, all_tokens)
    if dolist_result:
        dolist_var, list_ast, dolist_test_block, body_block, dolist_exit_block = dolist_result

        body_tokens = [
            t for t in all_tokens
            if body_block.start_offset <= parse_offset(t.offset) <= body_block.end_offset
        ]

        body_start_idx = 0
        for i in range(len(body_tokens)):
            if body_tokens[i].kind == 'VARSET' and body_tokens[i].attr == dolist_var:
                body_start_idx = i + 1
                break

        body_end_idx = len(body_tokens)
        for i in range(len(body_tokens) - 1, body_start_idx - 1, -1):
            if body_tokens[i].kind == 'VARREF' and body_tokens[i].attr == '--dolist-tail--':
                body_end_idx = i
                break

        actual_body_tokens = body_tokens[body_start_idx:body_end_idx]
        body_ast = bytecode_to_ast(actual_body_tokens)

        @dataclass
        class FilteredBodyNode:
            ast: object
            kind: str = "filtered_body"
            def to_ast(self, parser, all_tokens):
                return self.ast

        body_node = FilteredBodyNode(ast=body_ast)
        visited.add(body_block)
        return DolistNode(list_expr=list_ast, var_name=dolist_var, body_node=body_node)

    if has_backward_edge(entry_block):
        body_blocks = set()
        q = deque([entry_block])
        while q:
            cur = q.popleft()
            if cur in body_blocks:
                continue
            body_blocks.add(cur)
            for s in cur.successors:
                if s in region and s not in body_blocks:
                    q.append(s)
        header = entry_block
        exit_candidates = [s for b in body_blocks for s in b.successors if s not in body_blocks]
        exit_node = exit_candidates[0] if exit_candidates else None
        body_node = _structure_region(cfg, entry_block, exit_node, visited, all_tokens)
        return WhileNode(header_block=header, body_node=body_node)

    if is_conditional_header(entry_block):
        is_cond, cond_clauses = detect_cond_pattern(cfg, entry_block, region)

        if is_cond:
            structured_clauses = []
            for clause in cond_clauses:
                test_block = clause['test_block']
                then_block = clause['then_block']
                join = clause['join']

                then_node = _structure_region(cfg, then_block, join, visited, all_tokens) if then_block else None

                structured_clauses.append({
                    'test_block': test_block,
                    'then_block': then_block,
                    'then_node': then_node,
                    'join': join
                })

            return CondNode(clauses=structured_clauses)

        # Not a cond, check for regular if/when
        then_block = None
        else_block = None

        if getattr(entry_block, "follow_offset", None):
            then_block = cfg.block_offsets.get(entry_block.follow_offset)
        jump_offsets = getattr(entry_block, "jump_offsets", []) or []
        jump_offsets_list = sorted(jump_offsets) 
        else_block = cfg.block_offsets.get(jump_offsets_list[0]) if jump_offsets_list else None


        if then_block and else_block and then_block != else_block:
            join = find_common_successor(cfg, then_block, else_block)

            then_node = _structure_region(cfg, then_block, join, visited, all_tokens)
            else_node = _structure_region(cfg, else_block, join, visited, all_tokens)

            # Check if this is an 'or' pattern (GOTO-IF-NOT-NIL)
            # if is_or_pattern(entry_block):
            #     # In 'or' pattern: test GOTO-IF-NOT-NIL label
            #     # then_block is the join (when test is truthy)
            #     # else_block is the alternative (when test is nil)
            #     return OrNode(test_block=entry_block, alt_node=else_node, join_block=join)

         
            return IfNode(test_block=entry_block, then_node=then_node, else_node=else_node, join_block=join)

    parts = []
    cur = entry_block
    while cur and cur in region:
        if cur not in visited and cur != entry_block:
            visited.add(cur)
        if is_conditional_header(cur) and cur != entry_block:
            break
        if cur != entry_block and len([p for p in cur.predecessors if p in region]) > 1:
            break

        parts.append(BlockNode(block=cur))
        next_off = getattr(cur, "follow_offset", None)
        if next_off is None:
            break
        nxt = cfg.block_offsets.get(next_off)
        if not nxt or nxt not in region:
            break
        if len([p for p in nxt.predecessors if p in region]) > 1:
            break
        cur = nxt

    if len(parts) == 1:
        return parts[0]
    else:
        return SeqNode(parts=parts)


def parse_block_expressions(block, parser_factory):
    tokens = getattr(block, "tokens", None) or getattr(block, "instrs", None) or getattr(block, "instructions", None)
    if not tokens:
        return []
    parser = parser_factory(tokens)
    try:
        ast = parser.parse(tokens)
    except Exception as e:
        return [("PARSE-ERROR", block.start_offset, str(e))]

    return [ast]

def emit_ast(struct_node, parser_factory):
   
    if struct_node is None:
        return None
    if isinstance(struct_node, BlockNode):
        parsed = parse_block_expressions(struct_node.block, parser_factory)
        return {"type": "block", "block_offset": struct_node.block.start_offset, "body": parsed}
    if isinstance(struct_node, SeqNode):
        return {"type": "seq", "parts": [emit_ast(p, parser_factory) for p in struct_node.parts]}
    if isinstance(struct_node, IfNode):
        test_ast = emit_ast(BlockNode(block=struct_node.test_block), parser_factory)
        then_ast = emit_ast(struct_node.then_node, parser_factory)
        else_ast = emit_ast(struct_node.else_node, parser_factory) if struct_node.else_node else None
        return {"type": "if", "test": test_ast, "then": then_ast, "else": else_ast, "join": getattr(struct_node.join_block, "start_offset", None)}
    if isinstance(struct_node, WhileNode):
        header_ast = emit_ast(BlockNode(block=struct_node.header_block), parser_factory)
        body_ast = emit_ast(struct_node.body_node, parser_factory)
        return {"type": "while", "header": header_ast, "body": body_ast}
    return None
