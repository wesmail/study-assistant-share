"""The two errors a request can end in, shared by every module (review X1, Nov 2026).

    NotFound   what the request names doesn't exist: 404
    Refused    the request doesn't fit the current state; the message says why: 409

main.py turns them into those answers in one place; each module re-exports them, so
`annotations.Refused` and `library.Refused` are the same class.
"""


class NotFound(Exception):
    pass


class Refused(Exception):
    pass
