#!/usr/bin/env python3
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django; django.setup()
from colorama import Fore, Style, init as colorama_init
from tabulate import tabulate
from calls.models import CallSession
colorama_init(autoreset=True)

def list_calls(limit=10):
    rows = []
    for s in CallSession.objects.order_by("-started_at")[:limit]:
        rows.append([
            str(s.id)[:10]+"…", s.to_number, s.model_id or "—", s.status,
            s.total_turns, f"{s.avg_llm_latency_ms:.0f}ms",
            f"{s.avg_hallucination:.3f}" if s.avg_hallucination is not None else "—",
            f"{s.avg_faithfulness:.3f}"  if s.avg_faithfulness  is not None else "—",
            str(s.started_at)[:16],
        ])
    print(f"\n{Fore.CYAN}Recent Calls{Style.RESET_ALL}")
    print(tabulate(rows, headers=["ID","To","Model","Status","Turns","LLM lat","Hall.","Faith.","Started"], tablefmt="rounded_outline"))

def show_session(sid):
    try:
        s = CallSession.objects.get(id=sid)
    except:
        print(f"{Fore.RED}Not found{Style.RESET_ALL}"); return
    print(f"\n{Fore.CYAN}Session: {s.id} | Model: {s.model_id} | Status: {s.status}{Style.RESET_ALL}")
    for t in s.turns:
        print(f"\n{Fore.BLUE}Turn {t.turn_index+1}{Style.RESET_ALL}")
        print(f"  User: {t.user_input}")
        print(f"  AI  : {t.ai_response}")
        print(f"  LLM={t.llm_latency_ms:.0f}ms Total={t.total_latency_ms:.0f}ms Tokens={t.total_tokens}")
        if t.hallucination_score is not None:
            c = Fore.RED if t.hallucination_score > 0.5 else Fore.GREEN
            print(f"  {c}Hall={t.hallucination_score:.3f} Faith={t.faithfulness_score:.3f}{Style.RESET_ALL}")

def compare_models():
    from collections import defaultdict
    data = defaultdict(lambda: {"lat":[],"hall":[],"faith":[],"n":0})
    for s in CallSession.objects.all():
        k = s.model_id or "?"
        data[k]["n"] += 1
        if s.avg_llm_latency_ms:  data[k]["lat"].append(s.avg_llm_latency_ms)
        if s.avg_hallucination is not None: data[k]["hall"].append(s.avg_hallucination)
        if s.avg_faithfulness  is not None: data[k]["faith"].append(s.avg_faithfulness)
    avg = lambda l: f"{sum(l)/len(l):.2f}" if l else "—"
    rows = [[m,d["n"],avg(d["lat"])+"ms",avg(d["hall"]),avg(d["faith"])] for m,d in data.items()]
    print(f"\n{Fore.CYAN}Model Comparison{Style.RESET_ALL}")
    print(tabulate(rows, headers=["Model","Sessions","Avg LLM lat","Avg Hall.","Avg Faith."], tablefmt="rounded_outline"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session","-s")
    p.add_argument("--compare","-c", action="store_true")
    p.add_argument("--limit","-l", type=int, default=10)
    args = p.parse_args()
    if args.session: show_session(args.session)
    elif args.compare: compare_models()
    else: list_calls(args.limit)

if __name__ == "__main__":
    main()
