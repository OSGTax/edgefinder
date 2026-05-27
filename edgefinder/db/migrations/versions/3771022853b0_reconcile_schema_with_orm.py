"""reconcile schema with ORM

Revision ID: 3771022853b0
Revises: a1b2c3d4e5f6
Create Date: 2026-05-27 20:14:01.845926

Brings the migration chain back in line with the ORM, which had drifted
because prod was historically built via create_all rather than migrations:
  * adds tables ticker_dividends, ticker_news, ticker_splits, trade_context
  * adds the 8 enrichment/audit columns on trades the app writes
  * drops 4 dead scoring columns on fundamentals (referenced nowhere)
  * tightens daily_bars.source / created_at to NOT NULL (matching the ORM)

Batch operations are used so the migration runs on SQLite (dev) as well as
PostgreSQL (prod). JSON columns use the generic sa.JSON so they render as
JSON on both dialects.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '3771022853b0'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'ticker_dividends',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('ex_dividend_date', sa.String(length=20), nullable=False),
        sa.Column('pay_date', sa.String(length=20), nullable=True),
        sa.Column('cash_amount', sa.Float(), nullable=True),
        sa.Column('declaration_date', sa.String(length=20), nullable=True),
        sa.Column('frequency', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'ex_dividend_date', name='uq_ticker_dividend'),
    )
    op.create_index(op.f('ix_ticker_dividends_symbol'), 'ticker_dividends', ['symbol'], unique=False)
    op.create_table(
        'ticker_news',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('author', sa.String(length=200), nullable=True),
        sa.Column('published_utc', sa.String(length=30), nullable=True),
        sa.Column('article_url', sa.String(length=500), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('publisher_name', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'title', 'published_utc', name='uq_ticker_news'),
    )
    op.create_index('idx_ticker_news_symbol', 'ticker_news', ['symbol'], unique=False)
    op.create_table(
        'ticker_splits',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('execution_date', sa.String(length=20), nullable=False),
        sa.Column('split_from', sa.Integer(), nullable=True),
        sa.Column('split_to', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'execution_date', name='uq_ticker_split'),
    )
    op.create_index(op.f('ix_ticker_splits_symbol'), 'ticker_splits', ['symbol'], unique=False)
    op.create_table(
        'trade_context',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('trade_id', sa.String(length=36), nullable=False),
        sa.Column('recent_news', sa.JSON(), nullable=True),
        sa.Column('sector_prices', sa.JSON(), nullable=True),
        sa.Column('related_tickers', sa.JSON(), nullable=True),
        sa.Column('short_interest', sa.JSON(), nullable=True),
        sa.Column('dividends', sa.JSON(), nullable=True),
        sa.Column('indicators', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['trade_id'], ['trades.trade_id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('trade_id'),
    )

    with op.batch_alter_table('daily_bars') as batch_op:
        batch_op.alter_column('source', existing_type=sa.String(length=20), nullable=False,
                              existing_server_default=sa.text("'flatfiles'"))
        batch_op.alter_column('created_at', existing_type=sa.DateTime(), nullable=False,
                              existing_server_default=sa.text('(CURRENT_TIMESTAMP)'))

    with op.batch_alter_table('fundamentals') as batch_op:
        batch_op.drop_column('lynch_category')
        batch_op.drop_column('burry_score')
        batch_op.drop_column('lynch_score')
        batch_op.drop_column('composite_score')

    with op.batch_alter_table('trades') as batch_op:
        batch_op.add_column(sa.Column('entry_reasoning', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('exit_reasoning', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('indicators_at_entry', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('indicators_at_exit', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('fundamentals_at_entry', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('market_context_at_entry', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('pdt_flag', sa.Boolean(), server_default=sa.text('0'), nullable=False))
        batch_op.add_column(sa.Column('hold_duration_hours', sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('trades') as batch_op:
        batch_op.drop_column('hold_duration_hours')
        batch_op.drop_column('pdt_flag')
        batch_op.drop_column('market_context_at_entry')
        batch_op.drop_column('fundamentals_at_entry')
        batch_op.drop_column('indicators_at_exit')
        batch_op.drop_column('indicators_at_entry')
        batch_op.drop_column('exit_reasoning')
        batch_op.drop_column('entry_reasoning')

    with op.batch_alter_table('fundamentals') as batch_op:
        batch_op.add_column(sa.Column('composite_score', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('lynch_score', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('burry_score', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('lynch_category', sa.String(length=50), nullable=True))

    with op.batch_alter_table('daily_bars') as batch_op:
        batch_op.alter_column('created_at', existing_type=sa.DateTime(), nullable=True,
                              existing_server_default=sa.text('(CURRENT_TIMESTAMP)'))
        batch_op.alter_column('source', existing_type=sa.String(length=20), nullable=True,
                              existing_server_default=sa.text("'flatfiles'"))

    op.drop_table('trade_context')
    op.drop_index(op.f('ix_ticker_splits_symbol'), table_name='ticker_splits')
    op.drop_table('ticker_splits')
    op.drop_index('idx_ticker_news_symbol', table_name='ticker_news')
    op.drop_table('ticker_news')
    op.drop_index(op.f('ix_ticker_dividends_symbol'), table_name='ticker_dividends')
    op.drop_table('ticker_dividends')
