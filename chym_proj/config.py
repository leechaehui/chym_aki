# DB 연결
import sqlalchemy
DB_URL = "postgresql://bio4:bio4@localhost:5432/mimic4"
engine = sqlalchemy.create_engine(DB_URL)