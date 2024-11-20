from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# Import your models here, this allows Alembic to detect changes
from models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name)

# Set up metadata for `autogenerate`
target_metadata = Base.metadata

def run_migrations_online():
    connectable = config.attributes.get('connection', None)
    if connectable is None:
        from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
        connectable = create_async_engine(
            config.get_main_option("sqlalchemy.url"),
            echo=True,
            poolclass=pool.NullPool,
        )

    async def do_run_migrations(connection):
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
        )
        async with connection.begin():
            await context.run_migrations()

    from sqlalchemy.ext.asyncio import AsyncEngine
    if isinstance(connectable, AsyncEngine):
        import asyncio
        asyncio.run(do_run_migrations(connectable.connect()))
    else:
        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
            )
            with connection.begin():
                context.run_migrations()

run_migrations_online()
