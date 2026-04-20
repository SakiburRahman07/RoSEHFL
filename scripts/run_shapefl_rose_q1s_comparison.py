#!/usr/bin/env python3
"""
Run a strict effective-cost comparison between ShapeFL and ROSE-Q1S.
"""

from __future__ import annotations

from .run_shapefl_rose_effective_comparison import (
    build_parser as build_base_parser,
    run_effective_comparison,
)


def build_parser():
    parser = build_base_parser()
    parser.description = "ShapeFL vs ROSE-Q1S comparison with strict effective-cost accounting"
    parser.set_defaults(rose_strategy="rose_q1s")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.rose_strategy = "rose_q1s"
    run_effective_comparison(args)


if __name__ == "__main__":
    main()
