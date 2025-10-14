import os
import psycopg2
from tqdm import tqdm

# =======================
# 1️⃣ 数据库类定义
# =======================
class PostgresDB:
    """
    db = PostgresDB()
    query_sql = "select * from image"
    db.create_table(query_sql)
    """
    def __init__(self):
        """初始化数据库连接"""
        print("🔧 正在加载数据库配置...")
        self.db_config = {
            "host": os.getenv("DB_HOST"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_DATABASE"),
        }
        print("✅ 数据库配置加载成功")
        print("🔗 正在连接数据库...")
        self.conn = psycopg2.connect(**self.db_config)
        print("✅ 数据库连接成功")

    def query(self, sql: str, use_tqdm=False):
        """执行 SQL 查询，返回结果"""
        with self.conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            if use_tqdm:
                rows = [r for r in tqdm(rows, desc="查询进度", ncols=80)]
        print(f"✅ 查询完成，共返回 {len(rows)} 条记录")
        return rows

    def close(self):
        self.conn.close()
        print("🔒 数据库连接已关闭")