import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.shared.bases.base_model import ModelMixin
from config import Config


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='runtime.log'
)

logger = logging.getLogger('app')
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler('runtime.log'))

# app = FastAPI()
# app.add_middleware(
#     DBSessionMiddleware,
#     db_url=f"postgresql+psycopg2://{Config.postgres_connection}",
#     engine_args={"pool_size": 100000, "max_overflow": 10000},
# )
# with db():
#     ModelMixin.set_session(db.session)
engine = create_engine(f"postgresql+psycopg2://{Config.postgres_connection}", pool_size=100000, max_overflow=10000)
db = sessionmaker(bind=engine)
db.session = db()

with db.session:
    ModelMixin.set_session(db.session)


logger.info("Logging initialized")
