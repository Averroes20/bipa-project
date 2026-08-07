import asyncio
from sqlalchemy import MetaData
from app.core.database import engine
from sqlalchemy.orm import sessionmaker

def drop_all():
    print("Dropping all tables...")
    meta = MetaData()
    meta.reflect(bind=engine)
    meta.drop_all(bind=engine)
    # also drop alembic_version table if it exists
    with engine.connect() as conn:
        try:
            conn.execute("DROP TABLE alembic_version")
        except:
            pass
    print("All tables dropped.")

if __name__ == "__main__":
    drop_all()
