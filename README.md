# tmlc — a tiny machine learning compiler

`tmlc` is a small, from-scratch ML compiler/autograd system. The goal is to learn, but also to provide
a very modifiable, hackable frontend ML compiler. There are a lot of GPU kernel DSLs out there today,
each of which have their own pros and cons, but there's not many (to my knowledge) customizable frontends
that deal with the higher levels of the stack.

The idea behind TMLC is to provide a simple, strictly type-checked, and full-featured compiler that can
compile arbitrary computational graphs. The user-facing API may be slightly more tedious than other,
more established frameworks, but the hope is that the low level of abstraction and the tight typechecking
(thanks beartype!) will help both humans and agents understand exactly how any given piece of code works.

I try to manually maintain my README rather than have an agent do it, so this may lag a bit behind
development. However, I typically comment a lot in the codebase, and I've intentionally been type-annotating everything.
If this README doesn't cover a question, you should be able to just load the repo into Claude and ask questions!
Feel free to submit feature requests or ideas via issue tickets, but no guarantees things get looked at, and I'm
not going to accept unsolicited PRs at this time.

## Quickstart

See tests/logreg.py

## How it's put together

TODO

## What works right now

- Building a computational graph with broadcasting-aware arithmetic, matmul, exp/log/tanh,
  logsumexp, reshape/transpose/summation, and constants.
- Eager evaluation of any subset of the graph (`tmlc.run`), only computing the nodes actually
  needed for the requested outputs.
- Reverse-mode autodiff (`tmlc.gradients`) with the same "only touch what's needed" pruning.
- A logistic-regression test case (`tests/logreg.py`) that trains end-to-end and checks
  gradients against finite differences — the best example of a non-trivial graph working.

## What's not built yet

Low-level IR emission, an actual `compile()`/`run_compiled()` path, graph optimization passes (dead
code elimination, constant folding, CSE, op fusion), a real test suite (pytest isn't wired up —
`tests/*.py` are runnable scripts, not pytest files), and a pluggable array backend.
