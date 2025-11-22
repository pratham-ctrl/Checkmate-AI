#!/usr/bin/env python3
"""
Chess engine from scratch with AI (no external libs).

Features:
- Full rules: castling, en-passant, promotions
- Legal move filtering (no leaving king in check)
- Check/checkmate/stalemate detection
- Simple AI: negamax + alpha-beta with move ordering
- CLI for human vs human or human vs engine
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import time
import math
import sys

FILES = 'abcdefgh'
RANKS = '12345678'
WHITE, BLACK = 'w', 'b'
EMPTY = '.'

WHITE_PIECES = set('PNBRQK')
BLACK_PIECES = set('pnbrqk')

PIECE_VAL = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
    'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000
}

@dataclass
class Move:
    src: int
    dst: int
    promo: Optional[str] = None
    is_enpassant: bool = False
    is_castle: bool = False

def sq_to_coord(sq: int) -> str:
    f = sq % 8
    r = sq // 8
    return f"{FILES[f]}{8 - r}"

def coord_to_sq(coord: str) -> int:
    f = FILES.index(coord[0])
    r = 8 - int(coord[1])
    return r * 8 + f

def inside(r, c):
    return 0 <= r < 8 and 0 <= c < 8

class Board:
    def __init__(self, fen: str = None):
        # board is list of 64 chars (row-major, a8..h8, a7..h7, ..., a1..h1)
        self.board: List[str] = [EMPTY] * 64
        self.side_to_move: str = WHITE
        self.castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}
        self.ep_square: Optional[int] = None  # en-passant target square (square behind pawn that moved 2)
        self.halfmove = 0
        self.fullmove = 1
        if fen is None:
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.set_fen(fen)

    # ---------- FEN I/O ----------
    def set_fen(self, fen: str):
        parts = fen.split()
        board_part = parts[0]
        stm = parts[1]
        castling = parts[2]
        ep = parts[3]
        self.halfmove = int(parts[4]) if len(parts) > 4 else 0
        self.fullmove = int(parts[5]) if len(parts) > 5 else 1

        idx = 0
        for ch in board_part:
            if ch == '/':
                continue
            if ch.isdigit():
                n = int(ch)
                for _ in range(n):
                    self.board[idx] = EMPTY
                    idx += 1
            else:
                self.board[idx] = ch
                idx += 1

        self.side_to_move = WHITE if stm == 'w' else BLACK
        self.castling_rights = {
            'K': 'K' in castling,
            'Q': 'Q' in castling,
            'k': 'k' in castling,
            'q': 'q' in castling
        }
        self.ep_square = None if ep == '-' else coord_to_sq(ep)

    def get_fen(self) -> str:
        rows = []
        for r in range(8):
            empty = 0
            row = ''
            for f in range(8):
                p = self.board[r*8 + f]
                if p == EMPTY:
                    empty += 1
                else:
                    if empty:
                        row += str(empty); empty = 0
                    row += p
            if empty:
                row += str(empty)
            rows.append(row)
        board_part = '/'.join(rows)
        stm = 'w' if self.side_to_move == WHITE else 'b'
        cast = ''.join(k for k in ['K','Q','k','q'] if self.castling_rights[k]) or '-'
        ep = '-' if self.ep_square is None else sq_to_coord(self.ep_square)
        return f"{board_part} {stm} {cast} {ep} {self.halfmove} {self.fullmove}"

    # ---------- Display ----------
    def display(self):
        for r in range(8):
            rank_row = []
            for f in range(8):
                ch = self.board[r*8 + f]
                rank_row.append(ch if ch != EMPTY else '.')
            print(8 - r, ' '.join(rank_row))
        print('  ' + ' '.join(FILES))

    # ---------- Helpers ----------
    def color_of(self, p: str) -> Optional[str]:
        if p in WHITE_PIECES: return WHITE
        if p in BLACK_PIECES: return BLACK
        return None

    def king_square(self, color: str) -> int:
        target = 'K' if color == WHITE else 'k'
        for i, p in enumerate(self.board):
            if p == target:
                return i
        return -1

    def is_square_attacked(self, sq: int, by_color: str) -> bool:
        r, f = divmod(sq, 8)
        # Pawns
        if by_color == WHITE:
            for df in (-1, 1):
                rr, ff = r + 1, f + df
                if inside(rr, ff):
                    p = self.board[rr*8 + ff]
                    if p == 'P':
                        return True
        else:
            for df in (-1, 1):
                rr, ff = r - 1, f + df
                if inside(rr, ff):
                    p = self.board[rr*8 + ff]
                    if p == 'p':
                        return True
        # Knights
        for dr, df in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            rr, ff = r + dr, f + df
            if inside(rr, ff):
                p = self.board[rr*8 + ff]
                if by_color == WHITE and p == 'N': return True
                if by_color == BLACK and p == 'n': return True
        # Bishops / Queens diagonals
        for dr, df in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            rr, ff = r + dr, f + df
            while inside(rr, ff):
                p = self.board[rr*8 + ff]
                if p != EMPTY:
                    if by_color == WHITE and p in ('B','Q'): return True
                    if by_color == BLACK and p in ('b','q'): return True
                    break
                rr += dr; ff += df
        # Rooks / Queens straight
        for dr, df in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr, ff = r + dr, f + df
            while inside(rr, ff):
                p = self.board[rr*8 + ff]
                if p != EMPTY:
                    if by_color == WHITE and p in ('R','Q'): return True
                    if by_color == BLACK and p in ('r','q'): return True
                    break
                rr += dr; ff += df
        # King adjacency
        for dr in (-1,0,1):
            for df in (-1,0,1):
                if dr == 0 and df == 0: continue
                rr, ff = r + dr, f + df
                if inside(rr, ff):
                    p = self.board[rr*8 + ff]
                    if by_color == WHITE and p == 'K': return True
                    if by_color == BLACK and p == 'k': return True
        return False

    # ---------- Move generation (pseudo-legal) ----------
    def generate_pseudo_legal(self) -> List[Move]:
        moves: List[Move] = []
        stm = self.side_to_move
        for sq, p in enumerate(self.board):
            if p == EMPTY: continue
            color = self.color_of(p)
            if color != stm: continue
            r, f = divmod(sq, 8)
            enemy = BLACK_PIECES if color == WHITE else WHITE_PIECES
            # Pawn
            if p in ('P','p'):
                direction = -1 if p == 'P' else 1  # white pawns move up (decreasing row index)
                # single push
                rr, ff = r + direction, f
                if inside(rr, ff) and self.board[rr*8 + ff] == EMPTY:
                    dst = rr*8 + ff
                    # promotion?
                    if (p == 'P' and rr == 0) or (p == 'p' and rr == 7):
                        for promo in ('Q','R','B','N'):
                            moves.append(Move(sq, dst, promo if p == 'P' else promo.lower()))
                    else:
                        moves.append(Move(sq, dst))
                    # double push
                    start_row = 6 if p == 'P' else 1
                    rr2 = r + 2 * direction
                    if r == start_row and self.board[rr2*8 + ff] == EMPTY:
                        moves.append(Move(sq, rr2*8 + ff))
                # captures
                for df in (-1, 1):
                    rr, ff = r + direction, f + df
                    if inside(rr, ff):
                        dst = rr*8 + ff
                        tgt = self.board[dst]
                        if tgt != EMPTY and tgt in enemy:
                            if (p == 'P' and rr == 0) or (p == 'p' and rr == 7):
                                for promo in ('Q','R','B','N'):
                                    moves.append(Move(sq, dst, promo if p == 'P' else promo.lower()))
                            else:
                                moves.append(Move(sq, dst))
                # en-passant
                if self.ep_square is not None:
                    ep_r, ep_f = divmod(self.ep_square, 8)
                    if ep_r == r + direction and abs(ep_f - f) == 1:
                        moves.append(Move(sq, self.ep_square, is_enpassant=True))
            # Knight
            elif p in ('N','n'):
                for dr, df in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
                    rr, ff = r + dr, f + df
                    if inside(rr, ff):
                        dst = rr*8 + ff
                        tgt = self.board[dst]
                        if tgt == EMPTY or tgt in enemy:
                            moves.append(Move(sq, dst))
            # Bishop / Rook / Queen
            elif p in ('B','b','R','r','Q','q'):
                dirs = []
                if p in ('B','b'):
                    dirs = [(-1,-1),(-1,1),(1,-1),(1,1)]
                elif p in ('R','r'):
                    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
                else:
                    dirs = [(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)]
                for dr, df in dirs:
                    rr, ff = r + dr, f + df
                    while inside(rr, ff):
                        dst = rr*8 + ff
                        tgt = self.board[dst]
                        if tgt == EMPTY:
                            moves.append(Move(sq, dst))
                        else:
                            if tgt in enemy:
                                moves.append(Move(sq, dst))
                            break
                        rr += dr; ff += df
            # King
            elif p in ('K','k'):
                for dr in (-1,0,1):
                    for df in (-1,0,1):
                        if dr == 0 and df == 0: continue
                        rr, ff = r + dr, f + df
                        if inside(rr, ff):
                            dst = rr*8 + ff
                            tgt = self.board[dst]
                            if tgt == EMPTY or tgt in enemy:
                                moves.append(Move(sq, dst))
                # Castling (light checks: empties and not in/through check)
                if p == 'K' and self.side_to_move == WHITE:
                    if self.castling_rights['K']:
                        # e1->g1: f1,g1 empty (coords f1=5*? careful: row 7 index -> rank 1 is row 7)
                        if self.board[7*8+5] == EMPTY and self.board[7*8+6] == EMPTY:
                            if (not self.is_square_attacked(7*8+4, BLACK) and
                                not self.is_square_attacked(7*8+5, BLACK) and
                                not self.is_square_attacked(7*8+6, BLACK)):
                                moves.append(Move(7*8+4, 7*8+6, is_castle=True))
                    if self.castling_rights['Q']:
                        if self.board[7*8+3] == EMPTY and self.board[7*8+2] == EMPTY and self.board[7*8+1] == EMPTY:
                            if (not self.is_square_attacked(7*8+4, BLACK) and
                                not self.is_square_attacked(7*8+3, BLACK) and
                                not self.is_square_attacked(7*8+2, BLACK)):
                                moves.append(Move(7*8+4, 7*8+2, is_castle=True))
                if p == 'k' and self.side_to_move == BLACK:
                    if self.castling_rights['k']:
                        if self.board[0*8+5] == EMPTY and self.board[0*8+6] == EMPTY:
                            if (not self.is_square_attacked(0*8+4, WHITE) and
                                not self.is_square_attacked(0*8+5, WHITE) and
                                not self.is_square_attacked(0*8+6, WHITE)):
                                moves.append(Move(0*8+4, 0*8+6, is_castle=True))
                    if self.castling_rights['q']:
                        if self.board[0*8+3] == EMPTY and self.board[0*8+2] == EMPTY and self.board[0*8+1] == EMPTY:
                            if (not self.is_square_attacked(0*8+4, WHITE) and
                                not self.is_square_attacked(0*8+3, WHITE) and
                                not self.is_square_attacked(0*8+2, WHITE)):
                                moves.append(Move(0*8+4, 0*8+2, is_castle=True))
        return moves

    # ---------- Make / Unmake (we'll use copy approach for simplicity) ----------
    def copy(self) -> 'Board':
        b = Board(self.get_fen())
        return b

    def make_move(self, m: Move):
        p = self.board[m.src]
        q = self.board[m.dst]
        color = self.color_of(p)
        enemy = BLACK if color == WHITE else WHITE

        # halfmove clock
        if p in ('P','p') or q != EMPTY or m.is_enpassant:
            self.halfmove = 0
        else:
            self.halfmove += 1

        # reset ep
        self.ep_square = None

        # castling handling
        if m.is_castle and p in ('K','k'):
            if p == 'K':
                # white e1->g1 or e1->c1
                if m.dst == coord_to_sq('g1'):
                    self.board[7*8+4] = EMPTY
                    self.board[7*8+6] = 'K'
                    self.board[7*8+7] = EMPTY
                    self.board[7*8+5] = 'R'
                else:
                    self.board[7*8+4] = EMPTY
                    self.board[7*8+2] = 'K'
                    self.board[7*8+0] = EMPTY
                    self.board[7*8+3] = 'R'
                self.castling_rights['K'] = False
                self.castling_rights['Q'] = False
            else:
                if m.dst == coord_to_sq('g8'):
                    self.board[0*8+4] = EMPTY
                    self.board[0*8+6] = 'k'
                    self.board[0*8+7] = EMPTY
                    self.board[0*8+5] = 'r'
                else:
                    self.board[0*8+4] = EMPTY
                    self.board[0*8+2] = 'k'
                    self.board[0*8+0] = EMPTY
                    self.board[0*8+3] = 'r'
                self.castling_rights['k'] = False
                self.castling_rights['q'] = False
        else:
            # en-passant capture
            if m.is_enpassant and p in ('P','p'):
                self.board[m.dst] = p if m.promo is None else m.promo
                self.board[m.src] = EMPTY
                # remove captured pawn
                if p == 'P':
                    cap_sq = m.dst + 8
                else:
                    cap_sq = m.dst - 8
                self.board[cap_sq] = EMPTY
            else:
                # normal move (with possible promotion)
                self.board[m.dst] = p if m.promo is None else m.promo
                self.board[m.src] = EMPTY

            # update castle rights if king or rook moved / captured
            if p == 'K':
                self.castling_rights['K'] = False
                self.castling_rights['Q'] = False
            elif p == 'k':
                self.castling_rights['k'] = False
                self.castling_rights['q'] = False
            if p == 'R':
                if m.src == coord_to_sq('h1'):
                    self.castling_rights['K'] = False
                if m.src == coord_to_sq('a1'):
                    self.castling_rights['Q'] = False
            if p == 'r':
                if m.src == coord_to_sq('h8'):
                    self.castling_rights['k'] = False
                if m.src == coord_to_sq('a8'):
                    self.castling_rights['q'] = False
            if q == 'R':
                if m.dst == coord_to_sq('h1'):
                    self.castling_rights['K'] = False
                if m.dst == coord_to_sq('a1'):
                    self.castling_rights['Q'] = False
            if q == 'r':
                if m.dst == coord_to_sq('h8'):
                    self.castling_rights['k'] = False
                if m.dst == coord_to_sq('a8'):
                    self.castling_rights['q'] = False

            # set en-passant square for double pawn push
            if p == 'P':
                src_r = m.src // 8
                dst_r = m.dst // 8
                if src_r == 6 and dst_r == 4:
                    self.ep_square = m.src - 8
            elif p == 'p':
                src_r = m.src // 8
                dst_r = m.dst // 8
                if src_r == 1 and dst_r == 3:
                    self.ep_square = m.src + 8

        # switch side
        self.side_to_move = enemy
        if self.side_to_move == WHITE:
            self.fullmove += 1

    # ---------- Legal moves (filtering out king-in-check leaves) ----------
    def legal_moves(self) -> List[Move]:
        pseudo = self.generate_pseudo_legal()
        legal = []
        orig_side = self.side_to_move
        opp = BLACK if orig_side == WHITE else WHITE
        for m in pseudo:
            b2 = self.copy()
            b2.make_move(m)
            # find the king square of the side who just moved (orig_side)
            king_sq = b2.king_square(orig_side)
            if king_sq == -1:
                continue
            # If original side's king is attacked by opponent after move -> illegal
            if not b2.is_square_attacked(king_sq, opp):
                legal.append(m)
        return legal

    def in_check(self, color: str) -> bool:
        ks = self.king_square(color)
        if ks == -1:
            return False
        return self.is_square_attacked(ks, WHITE if color == BLACK else BLACK)

    def game_result(self) -> Optional[str]:
        if any(True for _ in self.legal_moves()):
            return None
        # no legal moves
        if self.in_check(self.side_to_move):
            # checkmate: the side to move is mated; winner is the opposite
            return f"checkmate: {'White' if self.side_to_move == BLACK else 'Black'} wins"
        else:
            return "stalemate"

    # ---------- Move parsing ----------
    def parse_move(self, s: str) -> Optional[Move]:
        s = s.strip()
        # castling variants
        if s in ("O-O", "0-0", "o-o"):
            return Move(coord_to_sq('e1'), coord_to_sq('g1'), is_castle=True) if self.side_to_move == WHITE else Move(coord_to_sq('e8'), coord_to_sq('g8'), is_castle=True)
        if s in ("O-O-O", "0-0-0", "o-o-o"):
            return Move(coord_to_sq('e1'), coord_to_sq('c1'), is_castle=True) if self.side_to_move == WHITE else Move(coord_to_sq('e8'), coord_to_sq('c8'), is_castle=True)
        # coordinates like e2e4 or e7e8q
        if len(s) in (4, 5) and s[0] in FILES and s[2] in FILES and s[1] in RANKS and s[3] in RANKS:
            src = coord_to_sq(s[:2])
            dst = coord_to_sq(s[2:4])
            promo = None
            if len(s) == 5:
                ch = s[4].lower()
                if ch in 'qrbn':
                    promo = ch.upper() if self.side_to_move == WHITE else ch
            # match against legal moves so flags (ep/castle/promo) are correct
            for m in self.legal_moves():
                if m.src == src and m.dst == dst:
                    if promo is not None:
                        # require promotion match
                        if m.promo is not None and m.promo.lower() == promo.lower():
                            return m
                        else:
                            continue
                    else:
                        if m.promo is not None:
                            # auto-promote to queen if user omitted (common behavior)
                            return Move(m.src, m.dst, 'Q' if self.side_to_move == WHITE else 'q', m.is_enpassant, m.is_castle)
                        return m
            return None
        return None

# ---------- Simple evaluation ----------
def evaluate(b: Board) -> int:
    # material only, positive = good for side-to-move
    s = 0
    for p in b.board:
        if p != EMPTY:
            s += PIECE_VAL.get(p, 0)
    # perspective: return score from side-to-move's point of view
    return s if b.side_to_move == WHITE else -s

# ---------- Move ordering ----------
def move_sort_key(b: Board, m: Move):
    tgt = b.board[m.dst]
    cap_value = 0
    if tgt != EMPTY:
        cap_value = abs(PIECE_VAL.get(tgt, 0))
    promo_value = abs(PIECE_VAL.get(m.promo.upper(), 0)) if m.promo else 0
    return (promo_value + cap_value, -abs(PIECE_VAL.get(b.board[m.src], 0)))

# ---------- Search (negamax + alpha-beta + quiescence) ----------
MATE = 9999999

def quiescence(b: Board, alpha: int, beta: int) -> int:
    stand = evaluate(b)
    if stand >= beta:
        return beta
    if alpha < stand:
        alpha = stand
    # only consider captures/promos
    captures = [m for m in b.legal_moves() if b.board[m.dst] != EMPTY or m.promo]
    captures.sort(key=lambda m: move_sort_key(b, m), reverse=True)
    for m in captures:
        b2 = b.copy()
        b2.make_move(m)
        score = -quiescence(b2, -beta, -alpha)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha

def negamax(b: Board, depth: int, alpha: int, beta: int) -> Tuple[int, Optional[Move]]:
    res = b.game_result()
    if res is not None:
        if res.startswith("checkmate"):
            # losing for side-to-move
            return (-MATE + 1, None)
        else:
            return (0, None)  # stalemate
    if depth == 0:
        return (quiescence(b, alpha, beta), None)
    best_score = -10**9
    best_move = None
    moves = b.legal_moves()
    moves.sort(key=lambda m: move_sort_key(b, m), reverse=True)
    for m in moves:
        b2 = b.copy()
        b2.make_move(m)
        score, _ = negamax(b2, depth - 1, -beta, -alpha)
        score = -score
        if score > best_score:
            best_score = score
            best_move = m
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best_score, best_move

def search_ai(b: Board, max_depth: int, time_limit_ms: int = 3000) -> Tuple[Optional[Move], int]:
    start = time.time()
    best_move = None
    best_score = -10**9
    for d in range(1, max_depth + 1):
        score, mv = negamax(b, d, -MATE, MATE)
        if mv is not None:
            best_move = mv
            best_score = score
        # time cutoff
        if (time.time() - start) * 1000 > time_limit_ms:
            break
    return best_move, best_score

# ---------- Perft (debug) ----------
def perft(b: Board, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for m in b.legal_moves():
        b2 = b.copy()
        b2.make_move(m)
        nodes += perft(b2, depth - 1)
    return nodes

# ---------- CLI ----------
def main():
    board = Board()
    engine_on = True
    engine_side = BLACK
    level = 3

    print("From-scratch Chess Engine with AI")
    print("Commands: e2e4, e7e8q, O-O, O-O-O, engine on/off, engine white/black, level N, perft N, fen, help, quit")
    while True:
        board.display()
        if board.in_check(board.side_to_move):
            print("(check)")
        gr = board.game_result()
        if gr:
            print(gr)
            break
        side_name = "White" if board.side_to_move == WHITE else "Black"

        # engine move?
        if engine_on and ((engine_side == WHITE and board.side_to_move == WHITE) or (engine_side == BLACK and board.side_to_move == BLACK)):
            print(f"Engine thinking (depth={level})...")
            mv, sc = search_ai(board, level, time_limit_ms=3000)
            if mv is None:
                print("Engine has no move.")
                break
            promo_str = '' if not mv.promo else mv.promo.lower()
            print(f"Engine plays {sq_to_coord(mv.src)}{sq_to_coord(mv.dst)}{promo_str}  (score {sc})")
            board.make_move(mv)
            continue

        cmd = input(f"{side_name} to move > ").strip()
        if not cmd:
            continue
        lc = cmd.lower()
        if lc in ('quit', 'q', 'exit'):
            break
        if lc == 'help':
            print("Moves examples: e2e4, h5f7, e7e8q (promotion), O-O, O-O-O")
            print("Commands: engine on/off | engine white/black | level N | perft N | fen | help | quit")
            continue
        if lc == 'fen':
            print(board.get_fen()); continue
        if lc.startswith('perft'):
            parts = lc.split()
            if len(parts) == 2 and parts[1].isdigit():
                d = int(parts[1])
                t0 = time.time()
                n = perft(board, d)
                t1 = time.time()
                print(f"perft({d}) = {n} nodes in {t1 - t0:.3f}s")
            else:
                print("Usage: perft N")
            continue
        if lc.startswith('engine'):
            parts = lc.split()
            if len(parts) == 2 and parts[1] in ('on', 'off'):
                engine_on = (parts[1] == 'on'); print("engine", "on" if engine_on else "off"); continue
            if len(parts) == 2 and parts[1] in ('white', 'black'):
                engine_side = WHITE if parts[1] == 'white' else BLACK
                print("engine side set to", parts[1]); continue
            print("Usage: engine on/off or engine white/black"); continue
        if lc.startswith('level'):
            parts = lc.split()
            if len(parts) == 2 and parts[1].isdigit():
                level = max(1, min(8, int(parts[1]))); print("level set to", level); continue
            print("Usage: level N"); continue

        mv = board.parse_move(cmd)
        if mv is None:
            print("Invalid move. Try again.")
            legals = board.legal_moves()
            if legals:
                print("Legal moves:", ' '.join(f"{sq_to_coord(m.src)}{sq_to_coord(m.dst)}{'' if not m.promo else m.promo.lower()}" for m in legals[:80]))
            continue
        board.make_move(mv)

if __name__ == '__main__':
    main()
