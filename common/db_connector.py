from sqlalchemy import create_engine
from common.config_loader import load_db_config


def get_db_engine():
    config = load_db_config()
    db_url = f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?charset=utf8mb4"

    try:
        engine = create_engine(db_url)
        return engine
    except Exception as e:
        print(f"DB 연결 에러: {e}")
        return None