"""A small interpreter for the csh README scripts CHARMM-GUI writes into gromacs/, openmm/ and namd/.

Nothing from the README is executed: `set`, `@`, `if`, `while`, `foreach`, backticks with printf/echo and `sed`
(on the text of the package files, in memory) are evaluated here, every other line becomes a `Command` with its
variables expanded, in the order the script would run it. The recipe then decides what each command means
(grompp + mdrun, openmm_run.py, namd2) and runs that itself, resumably. Pure: files are read through a callback."""
from __future__ import annotations

import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field


class CshError(ValueError):
    """The README uses something this interpreter does not understand."""


@dataclass
class Command:
    argv: list[str]
    line: int  # 1-based line in the README
    stdout: str | None = None  # `> file` or `>& file`
    stdin: str | None = None
    pipeline: list[list[str]] = field(default_factory=list)  # all stages when the line has `|`
    files: dict = field(default_factory=dict)  # files written earlier by sed/cp in this script, as they were then

    @property
    def text(self) -> str:
        out = " | ".join(" ".join(a) for a in (self.pipeline or [self.argv]))
        return out + (f" > {self.stdout}" if self.stdout else "")


@dataclass
class Result:
    commands: list[Command]
    variables: dict
    files: dict  # path -> text written by sed / cp (virtual, never on disk)
    notes: list[str]


MAX_ITERATIONS = 10000


# --- lines and blocks -----------------------------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    """Drop `# ...` when the # starts a word outside quotes (csh comment rule in scripts)."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i].rstrip()
    return line.rstrip()


def logical_lines(text: str) -> list[tuple[int, str]]:
    out, buf, start = [], "", 0
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        if not buf:
            start = n
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        full = _strip_comment(buf + line).strip()
        buf = ""
        if full:
            out.append((start, full))
    if buf.strip():
        out.append((start, _strip_comment(buf).strip()))
    return out


@dataclass
class Node:
    kind: str  # cmd, if, while, foreach
    line: int
    text: str = ""
    cond: str = ""
    var: str = ""
    body: list = field(default_factory=list)
    branches: list = field(default_factory=list)  # if: [(cond or None, [nodes])]


_IF_THEN = re.compile(r"^if\s*\((.*)\)\s*then$")
_ELSE_IF = re.compile(r"^else\s+if\s*\((.*)\)\s*then$")
_IF_ONE = re.compile(r"^if\s*\((.*?)\)\s*(\S.*)$")
_WHILE = re.compile(r"^while\s*\((.*)\)$")
_FOREACH = re.compile(r"^foreach\s+(\w+)\s*\((.*)\)$")


def parse_blocks(lines: list[tuple[int, str]]) -> list[Node]:
    pos = 0

    def block(stop: tuple) -> tuple[list[Node], str]:
        nonlocal pos
        nodes = []
        while pos < len(lines):
            n, s = lines[pos]
            word = s.split()[0]
            if word in stop or (s.startswith("else") and "else" in stop):
                return nodes, s
            pos += 1
            if m := _IF_THEN.match(s):
                node = Node("if", n)
                cond = m.group(1)
                while True:
                    body, end = block(("else", "endif"))
                    node.branches.append((cond, body))
                    pos += 1
                    if end == "endif":
                        break
                    if m2 := _ELSE_IF.match(end):
                        cond = m2.group(1)
                    elif end == "else":
                        cond = None
                    else:
                        raise CshError(f"linia {lines[pos - 1][0]}: nie rozumiem „{end}”")
                    if cond is None:
                        body, end = block(("endif",))
                        node.branches.append((None, body))
                        pos += 1
                        break
                nodes.append(node)
            elif m := _WHILE.match(s):
                body, _ = block(("end",))
                pos += 1
                nodes.append(Node("while", n, cond=m.group(1), body=body))
            elif m := _FOREACH.match(s):
                body, _ = block(("end",))
                pos += 1
                nodes.append(Node("foreach", n, var=m.group(1), text=m.group(2), body=body))
            elif m := _IF_ONE.match(s):
                nodes.append(Node("if", n, branches=[(m.group(1), [Node("cmd", n, text=m.group(2))])]))
            elif word in ("end", "endif", "else"):
                raise CshError(f"linia {n}: „{word}” bez początku bloku")
            else:
                nodes.append(Node("cmd", n, text=s))
        if stop:
            raise CshError(f"brak „{stop[-1]}” na końcu README")
        return nodes, ""

    nodes, _ = block(())
    return nodes


# --- expansion and expressions --------------------------------------------------------------------------------------

_VAR = re.compile(r"\$(\{(\w+)\}|(\w+)|\?(\w+))")


def expand(text: str, env: dict, line: int = 0) -> str:
    def sub(m):
        name = m.group(2) or m.group(3)
        if m.group(4):
            return "1" if m.group(4) in env else "0"
        if name not in env:
            raise CshError(f"linia {line}: zmienna ${name} nie jest ustawiona")
        return str(env[name])

    # single-quoted parts are literal in csh
    parts = re.split(r"('[^']*')", text)
    return "".join(p if p.startswith("'") else _VAR.sub(sub, p) for p in parts)


_TOKEN = re.compile(r"\s*(\d+|==|!=|<=|>=|&&|\|\||[<>()!+\-*/%]|\"[^\"]*\"|[^\s()<>=!&|+\-*/%]+)")


def evaluate(expr: str, line: int = 0):
    """csh @/if expression after variable expansion: integers, + - * / %, comparisons, && || !."""
    tokens, pos = [], 0
    expr = expr.strip()
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m:
            raise CshError(f"linia {line}: nie rozumiem wyrażenia „{expr}”")
        tokens.append(m.group(1).strip('"'))
        pos = m.end()
    i = 0

    def peek():
        return tokens[i] if i < len(tokens) else None

    def take():
        nonlocal i
        i += 1
        return tokens[i - 1]

    def atom():
        t = take() if peek() is not None else None
        if t is None:
            raise CshError(f"linia {line}: niepełne wyrażenie „{expr}”")
        if t == "(":
            v = disj()
            if take() != ")":
                raise CshError(f"linia {line}: brak „)” w „{expr}”")
            return v
        if t == "!":
            return int(not truth(atom()))
        if t == "-":
            return -num(atom())
        return int(t) if re.fullmatch(r"-?\d+", t) else t

    def num(v):
        if isinstance(v, int):
            return v
        raise CshError(f"linia {line}: „{v}” nie jest liczbą")

    def truth(v):
        return v != 0 and v != "" and v != "0"

    def term():
        v = atom()
        while peek() in ("*", "/", "%"):
            op, w = take(), num(atom())
            v = num(v) * w if op == "*" else (int(num(v) / w) if op == "/" else num(v) % w)
        return v

    def arith():
        v = term()
        while peek() in ("+", "-"):
            op, w = take(), num(term())
            v = num(v) + w if op == "+" else num(v) - w
        return v

    def comp():
        v = arith()
        while peek() in ("==", "!=", "<", ">", "<=", ">="):
            op, w = take(), arith()
            if op in ("==", "!="):
                same = str(v) == str(w)
                v = int(same if op == "==" else not same)
            else:
                a, b = num(v), num(w)
                v = int({"<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b}[op])
        return v

    def conj():
        v = comp()
        while peek() == "&&":
            take()
            w = comp()
            v = int(truth(v) and truth(w))
        return v

    def disj():
        v = conj()
        while peek() == "||":
            take()
            w = conj()
            v = int(truth(v) or truth(w))
        return v

    value = disj()
    if peek() is not None:
        raise CshError(f"linia {line}: nadmiarowe „{peek()}” w „{expr}”")
    return value


def _truth(v) -> bool:
    return v != 0 and v != "" and v != "0"


# --- the interpreter ------------------------------------------------------------------------------------------------

def _printf(args: list[str]) -> str:
    if not args:
        return ""
    fmt, values = args[0], args[1:]
    conv = []
    specs = re.findall(r"%[-+ 0#]*\d*(?:\.\d+)?([diouxXeEfgGs])", fmt)
    for spec, v in zip(specs, values, strict=False):
        conv.append(int(v) if spec in "diouxX" else (float(v) if spec in "eEfgG" else v))
    try:
        return fmt.replace("\\n", "\n") % tuple(conv)
    except (TypeError, ValueError) as exc:
        raise CshError(f"printf {fmt}: {exc}") from None


def _sed_expr(expr: str) -> tuple[re.Pattern, str, bool]:
    if len(expr) < 4 or expr[0] != "s":
        raise CshError(f"sed: obsługuję tylko s/…/…/, a jest „{expr}”")
    d = expr[1]
    parts = re.split(r"(?<!\\)" + re.escape(d), expr[2:])
    if len(parts) < 3:
        raise CshError(f"sed: niepełne wyrażenie „{expr}”")
    pat, repl, flags = parts[0], parts[1], parts[2]
    # basic regular expressions: \( \) \{ \} are groups/intervals, bare ones are literal
    pat = re.sub(r"\\([(){}|+?])|([(){}|+?])", lambda m: m.group(1) or "\\" + m.group(2), pat)
    repl = repl.replace("\\" + d, d)
    repl = re.sub(r"(?<!\\)&", r"\\g<0>", repl)
    return re.compile(pat), repl, "g" in flags


def sed(args: list[str], text: str) -> str:
    """`sed [-e] 's/a/b/[g]' ...` on text (the only sed CHARMM-GUI READMEs use)."""
    exprs, i = [], 0
    while i < len(args):
        if args[i] == "-e":
            exprs.append(args[i + 1])
            i += 2
        elif args[i].startswith("-"):
            raise CshError(f"sed: opcja {args[i]} nieobsługiwana")
        else:
            exprs.append(args[i])
            i += 1
    compiled = [_sed_expr(e) for e in exprs]
    out = []
    for line in text.split("\n"):
        for pat, repl, glob in compiled:
            line = pat.sub(repl, line, count=0 if glob else 1)
        out.append(line)
    return "\n".join(out)


def _split(text: str, line: int) -> list[str]:
    lex = shlex.shlex(text, posix=True, punctuation_chars="|<>&;")
    lex.whitespace_split = True
    try:
        return list(lex)
    except ValueError as exc:
        raise CshError(f"linia {line}: {exc}") from None


class Interpreter:
    def __init__(self, read_file: Callable[[str], str | None], force: dict | None = None):
        self.read_file = read_file
        self.force = dict(force or {})  # variables pinned to a value (e.g. cntmax from params)
        self.env: dict = {}
        self.files: dict = {}
        self.commands: list[Command] = []
        self.notes: list[str] = []
        self.steps = 0

    def read(self, path: str, line: int) -> str:
        if path in self.files:
            return self.files[path]
        text = self.read_file(path)
        if text is None:
            raise CshError(f"linia {line}: README czyta {path}, a tego pliku nie ma w paczce")
        return text

    def assign(self, name: str, value):
        self.env[name] = self.force.get(name, value)

    def backticks(self, text: str, line: int) -> str:
        def sub(m):
            argv = _split(m.group(1), line)
            if not argv:
                return ""
            if argv[0] == "printf":
                return _printf(argv[1:])
            if argv[0] == "echo":
                return " ".join(argv[1:])
            raise CshError(f"linia {line}: nie wykonuję `{m.group(1)}` (tylko printf i echo)")

        return re.sub(r"`([^`]*)`", sub, text)

    def run(self, nodes: list[Node]):
        for node in nodes:
            self.steps += 1
            if self.steps > MAX_ITERATIONS * 10:
                raise CshError("README kręci się w pętli bez końca")
            if node.kind == "cmd":
                self.command(node.text, node.line)
            elif node.kind == "if":
                for cond, body in node.branches:
                    if cond is None or _truth(evaluate(expand(cond, self.env, node.line), node.line)):
                        self.run(body)
                        break
            elif node.kind == "while":
                n = 0
                while _truth(evaluate(expand(node.cond, self.env, node.line), node.line)):
                    n += 1
                    if n > MAX_ITERATIONS:
                        raise CshError(f"linia {node.line}: pętla while bez końca")
                    self.run(node.body)
            elif node.kind == "foreach":
                for value in _split(self.backticks(expand(node.text, self.env, node.line), node.line), node.line):
                    self.assign(node.var, value)
                    self.run(node.body)

    def command(self, text: str, line: int):
        word = text.split()[0]
        if word == "set":
            m = re.match(r"set\s+(\w+)\s*=\s*(.*)$", text)
            if not m:
                if re.fullmatch(r"set\s+\w+", text):
                    self.assign(text.split()[1], "")
                    return
                raise CshError(f"linia {line}: nie rozumiem „{text}”")
            value = self.backticks(expand(m.group(2), self.env, line), line).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            self.assign(m.group(1), value)
            return
        if word in ("setenv",):
            bits = text.split(None, 2)
            self.assign(bits[1], expand(bits[2], self.env, line) if len(bits) > 2 else "")
            return
        if word == "@" or text.startswith("@"):
            m = re.match(r"@\s*(\w+)\s*(\+\+|--|[+\-*/]?=)\s*(.*)$", text)
            if not m:
                raise CshError(f"linia {line}: nie rozumiem „{text}”")
            name, op, rhs = m.groups()
            cur = int(self.env.get(name, 0)) if op != "=" else 0
            if op == "++":
                value = cur + 1
            elif op == "--":
                value = cur - 1
            else:
                v = evaluate(expand(rhs, self.env, line), line)
                if not isinstance(v, int):
                    raise CshError(f"linia {line}: „{rhs}” nie jest liczbą")
                value = {"=": v, "+=": cur + v, "-=": cur - v, "*=": cur * v, "/=": int(cur / v) if v else 0}[op]
            self.assign(name, value)
            return
        if word in ("unset", "exit", "endif", "shift"):
            return
        tokens = _split(self.backticks(expand(text, self.env, line), line), line)
        stages, cur, stdout, stdin = [], [], None, None
        it = iter(range(len(tokens)))
        for i in it:
            t = tokens[i]
            if t == "|":
                stages.append(cur)
                cur = []
            elif t in (">", ">>", ">&", ">>&", ">!", ">&!"):
                stdout = tokens[i + 1] if i + 1 < len(tokens) else None
                next(it, None)
            elif t == "<":
                stdin = tokens[i + 1] if i + 1 < len(tokens) else None
                next(it, None)
            elif t in ("&", ";"):
                continue
            else:
                cur.append(t)
        stages.append(cur)
        stages = [s for s in stages if s]
        if not stages:
            return
        if all(s[0] in ("sed", "cat") for s in stages) and stdout:
            self.text_pipeline(stages, stdout, stdin, line)
            return
        if stages[0][0] == "cp" and len(stages[0]) == 3:
            self.files[stages[0][2]] = self.read(stages[0][1], line)
            return
        if stages[0][0] in ("echo", "cd", "mkdir", "rm", "date", "sleep", "source"):
            if stages[0][0] in ("cd", "source"):
                self.notes.append(f"linia {line}: pomijam „{' '.join(stages[0])}”")
            return
        cmd = Command(stages[0], line, stdout=stdout, stdin=stdin, pipeline=stages if len(stages) > 1 else [])
        cmd.files = {a: self.files[a] for s in stages for a in s if a in self.files}
        self.commands.append(cmd)

    def text_pipeline(self, stages, stdout, stdin, line):
        text = None if stdin is None else self.read(stdin, line)
        for argv in stages:
            args = argv[1:]
            if argv[0] == "cat":
                text = "".join(self.read(a, line) for a in args) if args else text
                continue
            # sed -e EXPR [-e EXPR] [FILE...]  or  sed EXPR [FILE...]
            exprs, files, i = [], [], 0
            with_e = "-e" in args
            while i < len(args):
                if args[i] == "-e" and i + 1 < len(args):
                    exprs += ["-e", args[i + 1]]
                    i += 2
                    continue
                if not with_e and not exprs:
                    exprs.append(args[i])
                else:
                    files.append(args[i])
                i += 1
            src = "".join(self.read(f, line) for f in files) if files else text
            if src is None:
                raise CshError(f"linia {line}: sed bez pliku wejściowego")
            text = sed(exprs, src)
        self.files[stdout] = text or ""


def interpret(text: str, read_file: Callable[[str], str | None], force: dict | None = None) -> Result:
    interp = Interpreter(read_file, force)
    interp.run(parse_blocks(logical_lines(text)))
    return Result(interp.commands, interp.env, interp.files, interp.notes)
