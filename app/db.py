"""SQLite persistence: connection management, schema and transactions."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

# The database path is resolved lazily so tests can override it before the
# first connection is opened.
DB_PATH: str = os.environ.get("SUBNET_DB_PATH", "/data/subnets.db")

# SQLite returns SQLITE_BUSY immediately when another connection holds the
# write lock; wait this long before giving up.
BUSY_TIMEOUT_SECONDS = 10.0

# Confirmable BGP announcement plans.  Every row is immutable after
# creation: only ``status`` moves 'planned' -> 'active' (single activation
# or handover) and 'active' -> 'retired' (handover).  Deleting the pool
# cascades to its plans.  Kept as a constant so the schema migration below
# can rebuild the table with the identical definition.
_BGP_PLANS_DDL = """
            CREATE TABLE IF NOT EXISTS bgp_plans (
                plan_id          TEXT    PRIMARY KEY,
                pool_id          INTEGER NOT NULL REFERENCES pools(id) ON DELETE CASCADE,
                params           TEXT    NOT NULL,
                audit_results    TEXT    NOT NULL,
                pool_state       TEXT    NOT NULL,
                state_digest     TEXT    NOT NULL,
                status           TEXT    NOT NULL CHECK (status IN ('planned', 'active', 'retired')),
                created_at       TEXT    NOT NULL,
                activated_at     TEXT
            );
"""

# One row per completed one-shot address provisioning.  Kept as a constant so
# the schema migration below can rebuild the table with the identical
# definition.  See the matching comment inside init_db for why allocation_id
# is deliberately a plain historical reference rather than an FK.
_ADDRESS_PROVISIONS_DDL = """
            CREATE TABLE IF NOT EXISTS address_provisions (
                plan_id        TEXT    PRIMARY KEY
                                  REFERENCES bgp_plans(plan_id) ON DELETE CASCADE,
                pool_id        INTEGER NOT NULL REFERENCES pools(id) ON DELETE CASCADE,
                allocation_id  INTEGER NOT NULL,
                request_id     TEXT    NOT NULL,
                params         TEXT    NOT NULL,
                result         TEXT    NOT NULL,
                created_at     TEXT    NOT NULL
            );
"""


def _legacy_bgp_plans(conn: sqlite3.Connection) -> bool:
    """Whether ``bgp_plans`` predates the ``retired`` status."""

    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'bgp_plans'"
    ).fetchone()
    return row is not None and "'retired'" not in row["sql"]


def _legacy_address_provisions(conn: sqlite3.Connection) -> bool:
    """Whether ``address_provisions.allocation_id`` still cascades.

    The new definition has no REFERENCES clause on ``allocation_id`` at all.
    """

    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'address_provisions'"
    ).fetchone()
    return row is not None and "REFERENCES allocations" in row["sql"]


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        is not None
    )


def _migrate_legacy_tables(conn: sqlite3.Connection) -> None:
    """Rebuild the legacy ``bgp_plans`` / ``address_provisions`` tables together.

    The two old schemas cannot be rebuilt independently: the legacy
    ``address_provisions`` declared ``plan_id ... REFERENCES bgp_plans(plan_id)
    ON DELETE CASCADE`` and ``allocation_id INTEGER NOT NULL REFERENCES
    allocations(id) ON DELETE CASCADE``, so dropping a renamed-away legacy
    ``bgp_plans`` while foreign keys are enforced cascades every provision
    record away.  An upgrade on a database carrying an un-released provision
    therefore lost the linkage row: the allocation and plan survived, and the
    identical request could no longer replay -- it failed with
    ``request_id_conflict`` against the surviving allocation, and once the
    allocation was released a retry hit ``bgp_plan_id_conflict`` against the
    surviving plan.

    Everything that needs rebuilding is therefore rebuilt in **one**
    transaction with foreign-key enforcement suspended for its duration (a
    ``PRAGMA foreign_keys`` change is only effective outside a transaction):
    old tables renamed away, new tables created, every row copied verbatim
    (each keeps its primary-key identity and the plan/allocation/pool links)
    and the legacy tables dropped before the commit.  Any failure rolls the
    whole batch back -- no table ends up half-rebuilt -- and a repeated run
    simply performs the still-pending rebuild.

    Rebuilding ``bgp_plans`` also forces a rebuild of ``address_provisions``
    when that table exists, even if its own schema is already current: the
    rename of the referenced table would otherwise leave the provision table
    dangling on the soon-dropped legacy name.

    A copy whose ``allocation_id`` points at an already-released allocation is
    expected and kept: the rebuilt table treats it as a non-FK historical
    reference precisely so such a release stays replayable.
    """

    plans_legacy = _legacy_bgp_plans(conn)
    provisions_legacy = _legacy_address_provisions(conn)
    if not plans_legacy and not provisions_legacy:
        return  # fresh database (created below) or already migrated

    provisions_exist = _table_exists(conn, "address_provisions")
    rebuild_plans = plans_legacy
    rebuild_provisions = provisions_legacy or (plans_legacy and provisions_exist)

    # Foreign-key enforcement is per connection and may only be toggled while
    # no transaction is open; init_db owns this short-lived connection, so the
    # suspension never leaks to application traffic.
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            if rebuild_plans:
                conn.execute("ALTER TABLE bgp_plans RENAME TO bgp_plans_legacy")
                conn.execute(_BGP_PLANS_DDL)
                conn.execute(
                    "INSERT INTO bgp_plans (plan_id, pool_id, params, audit_results, "
                    "pool_state, state_digest, status, created_at, activated_at) "
                    "SELECT plan_id, pool_id, params, audit_results, pool_state, "
                    "state_digest, status, created_at, activated_at "
                    "FROM bgp_plans_legacy"
                )
            if rebuild_provisions:
                # With the (possibly also renamed) plans table out of the way
                # the old table cannot cascade on anything during the rebuild.
                conn.execute(
                    "ALTER TABLE address_provisions RENAME TO address_provisions_legacy"
                )
                conn.execute(_ADDRESS_PROVISIONS_DDL)
                conn.execute(
                    "INSERT INTO address_provisions (plan_id, pool_id, allocation_id, "
                    "request_id, params, result, created_at) "
                    "SELECT plan_id, pool_id, allocation_id, request_id, params, result, "
                    "created_at FROM address_provisions_legacy"
                )
            # No PRAGMA foreign_key_check here on purpose: an old database that
            # released a provisioned allocation kept the bgp_plans row while the
            # legacy cascading allocation_id FK had already deleted the provision
            # row (exactly the historical state the new table tolerates).  Every
            # surviving copy -- the only kind that reaches this code -- resolves
            # under the rebuilt schema, whose allocation_id is a deliberately
            # non-FK historical reference; an allocation_id that points at a
            # deleted allocation is a valid replay anchor, not a violation.
            #
            # Dropping the legacy tables also drops their indexes (notably
            # idx_bgp_plans_pool / idx_address_provisions_request); the
            # CREATE INDEX statements in init_db recreate them on the new
            # tables.  Order is irrelevant with enforcement suspended.
            if rebuild_provisions:
                conn.execute("DROP TABLE address_provisions_legacy")
            if rebuild_plans:
                conn.execute("DROP TABLE bgp_plans_legacy")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def _connect() -> sqlite3.Connection:
    directory = os.path.dirname(os.path.abspath(DB_PATH))
    os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(
        DB_PATH,
        timeout=BUSY_TIMEOUT_SECONDS,
        isolation_level=None,  # autocommit; transactions are managed explicitly
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_SECONDS * 1000)}")
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL lets readers run while a writer holds the lock; allocation itself
    # is still serialised by the BEGIN IMMEDIATE write transaction.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = FULL")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        _migrate_legacy_tables(conn)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS pools (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                cidr        TEXT    NOT NULL UNIQUE,
                created_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS exclusions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                pool_id     INTEGER NOT NULL REFERENCES pools(id) ON DELETE CASCADE,
                cidr        TEXT    NOT NULL,
                created_at  TEXT    NOT NULL,
                UNIQUE (pool_id, cidr)
            );

            CREATE TABLE IF NOT EXISTS allocations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                pool_id         INTEGER NOT NULL REFERENCES pools(id) ON DELETE CASCADE,
                cidr            TEXT    NOT NULL,
                request_id      TEXT    NOT NULL UNIQUE,
                prefix_length   INTEGER NOT NULL,
                created_at      TEXT    NOT NULL
            );

            -- One row per completed restore; the primary key makes a repeated
            -- restore_id replay the original result instead of restoring twice.
            CREATE TABLE IF NOT EXISTS snapshot_restores (
                restore_id  TEXT    PRIMARY KEY,
                pool_id     INTEGER NOT NULL REFERENCES pools(id) ON DELETE CASCADE,
                digest      TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
            );

            -- One row per completed pool partition.  The UNIQUE operation_id
            -- makes a retried partition replay the original result instead of
            -- splitting twice.  source_pool_id is a plain historical reference:
            -- the migration itself deletes the source pool row, so no FK.
            CREATE TABLE IF NOT EXISTS pool_migrations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id    TEXT    NOT NULL UNIQUE,
                source_pool_id  INTEGER NOT NULL,
                source_cidr     TEXT    NOT NULL,
                sub_pools       TEXT    NOT NULL,
                created_at      TEXT    NOT NULL
            );

            -- Confirmable BGP announcement plans; see _BGP_PLANS_DDL above.
            -- The migration in init_db rebuilds this table for databases
            -- created before the 'retired' status existed.
            """
            + _BGP_PLANS_DDL
            + """

            -- One row per completed policy-gated batch reservation.  The
            -- batch_id primary key makes a retried batch replay the original
            -- result instead of reserving twice; params is the canonical
            -- request fingerprint used to detect conflicting reuses.
            CREATE TABLE IF NOT EXISTS reservation_batches (
                batch_id    TEXT    PRIMARY KEY,
                pool_id     INTEGER NOT NULL REFERENCES pools(id) ON DELETE CASCADE,
                params      TEXT    NOT NULL,
                result      TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
            );

            -- One row per completed cross-pool allocation migration.  The
            -- UNIQUE operation_id makes a retried migration replay the
            -- original result instead of moving the allocations twice.
            -- source_pool_id is a plain historical reference: the source
            -- pool of a later partition migration is deleted, so no FK;
            -- target_pool_id keeps the cascade so deleting the destination
            -- pool removes the record.  params is the canonical request
            -- fingerprint (manifest order is significant) and result holds
            -- the stored input-order mapping.
            CREATE TABLE IF NOT EXISTS allocation_migrations (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id     TEXT    NOT NULL UNIQUE,
                source_pool_id   INTEGER NOT NULL,
                target_pool_id   INTEGER NOT NULL REFERENCES pools(id) ON DELETE CASCADE,
                source_cidr      TEXT    NOT NULL,
                target_cidr      TEXT    NOT NULL,
                params           TEXT    NOT NULL,
                result           TEXT    NOT NULL,
                created_at       TEXT    NOT NULL
            );

            -- One row per completed one-shot address provisioning.  Ties the
            -- regular allocation to the BGP announcement plan stored together
            -- with it in a single write transaction.  plan_id is both the
            -- idempotency key of the provision and an FK to the bgp_plans row
            -- it created, so a plan_id taken by a standalone plan (or by a
            -- previous provision) cannot be taken over here.  Deleting the
            -- linked plan or pool cascades the provision record away.
            --
            -- allocation_id is intentionally a plain historical reference,
            -- NOT a foreign key: releasing the provisioned allocation
            -- (DELETE /pools/{id}/allocations/{request_id}) must not delete
            -- this record.  The provision keeps being replayable by plan_id --
            -- it returns the stored original allocation, announcement and
            -- plan (with the plan's current status) instead of re-occupying
            -- the released address or colliding with a plan_id that is still
            -- occupied -- even if the freed request_id is later reused by an
            -- ordinary allocation.  The schema migration in init_db rebuilds
            -- older databases whose allocation_id carried ON DELETE CASCADE.
            """
            + _ADDRESS_PROVISIONS_DDL
            + """

            -- Fast lookup of all plans belonging to one pool.
            CREATE INDEX IF NOT EXISTS idx_allocations_pool
                ON allocations(pool_id);
            CREATE INDEX IF NOT EXISTS idx_exclusions_pool
                ON exclusions(pool_id);
            CREATE INDEX IF NOT EXISTS idx_bgp_plans_pool
                ON bgp_plans(pool_id);
            -- Fast takeover lookup by request_id for one-shot provisions.
            CREATE INDEX IF NOT EXISTS idx_address_provisions_request
                ON address_provisions(request_id);
            """
        )


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def write_transaction() -> Iterator[sqlite3.Connection]:
    """Open an immediate write transaction.

    ``BEGIN IMMEDIATE`` acquires the reserved write lock up-front, so two
    concurrent allocations are serialised: the second one waits (up to
    ``busy_timeout``) instead of failing on commit.
    """

    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


@contextmanager
def read_transaction() -> Iterator[sqlite3.Connection]:
    """Open a deferred (read) transaction for a consistent snapshot."""

    conn = _connect()
    try:
        conn.execute("BEGIN")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
