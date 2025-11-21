
import oracledb
import numpy as np
import os
import array
from dotenv import load_dotenv

# env
load_dotenv() 

# thick client
oracledb.init_oracle_client(lib_dir=r"D:\\instantclient_23_9")

# Conectar ao Oracle
def connect_database():
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    dsn = os.getenv("DB_URL")

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        print("Connection successful!")
    except Exception as e:
        print("Connection failed!")
    
    return connection

def read_fvecs(filename):
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
        dim = data[0]
        return data.reshape(-1, dim + 1)[:, 1:].astype(np.float32)

import array

import array
import time
import numpy as np

def vector_search_in_oracle(vectors, k=100):
    conn = connect_database()
    cursor = conn.cursor()
    search_sql = """
        SELECT id, embedding, VECTOR_DISTANCE(embedding, :1, EUCLIDEAN_SQUARED) AS distance
        FROM vector_table
        ORDER BY distance
        FETCH FIRST :2 ROWS ONLY
    """

    timings = []

    for idx, vec in enumerate(vectors):
        arr_vec = array.array('f', vec)
        start = time.perf_counter()
        cursor.execute(search_sql, (arr_vec, k))
        results = cursor.fetchall()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

        print(f"🔎 Resultados para vetor {idx} (top {k}):")
        for row in results:
            print(f"  → id={row[0]}, distance={row[2]}")
        print(f"⏱️ Tempo consulta: {elapsed:.4f} segundos\n")

    # Estatísticas de tempo
    np_timings = np.array(timings)
    media = np.mean(np_timings)
    minimo = np.min(np_timings)
    maximo = np.max(np_timings)
    p95 = np.percentile(np_timings, 95)
    p99 = np.percentile(np_timings, 99)

    print("===== Estatísticas das buscas =====")
    print(f"Média     : {media:.4f} s")
    print(f"Mínimo    : {minimo:.4f} s")
    print(f"Máximo    : {maximo:.4f} s")
    print(f"Percentil 95: {p95:.4f} s")
    print(f"Percentil 99: {p99:.4f} s")

    cursor.close()
    conn.close()

def main():
    # Caminho para o arquivo .fvecs
    file_path = 'dataset/sift_query.fvecs'

    try:
        # Ler os vetores do arquivo .fvecs
        vectors = read_fvecs(file_path)
        print("read_fvecs ok")        

        # vector search
        vector_search_in_oracle(vectors)           

    except Exception as e:
        print(f"Erro ao processar os vetores: {e}")            

if __name__ == "__main__":
    main()
