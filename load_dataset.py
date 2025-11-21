"""
Change Instant Client diretory (see below - thick client)

Database setup commands before loading:

DROP TABLE vector_table PURGE;
CREATE TABLE vector_table (id INT, embedding VECTOR);

set serveroutput on
variable response_json clob;
begin
DBMS_VECTOR.INDEX_VECTOR_MEMORY_ADVISOR(
    INDEX_TYPE=>'HNSW', 
    NUM_VECTORS=>1000000, 
    DIM_COUNT=>128, 
    DIM_TYPE=>'FLOAT32',  
    RESPONSE_JSON=>:response_json); 
end;
/

-- after loading data
CREATE VECTOR INDEX vector_index
ON vector_table (embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH DISTANCE EUCLIDEAN_SQUARED
PARALLEL 4;
"""

import oracledb
import numpy as np
import os
import array
from dotenv import load_dotenv

# env
load_dotenv() 

# thick client
oracledb.init_oracle_client(lib_dir=r"D:\\instantclient_23_9") # change

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

def insert_vectors_to_oracle(vectors):
    # Inserir os vetores em lotes
    insert_sql = "INSERT INTO vector_table (id, embedding) VALUES (:1, :2)"
    batch_size = 500
    conn = connect_database()
    cursor = conn.cursor()
    
    print("⬆️ Inserindo vetores no Oracle...")
    for i in range(0, len(vectors), batch_size):
        batch = [(int(j), array.array('f', vectors[j])) for j in range(i, min(i + batch_size, len(vectors)))]
        cursor.executemany(insert_sql, batch)
        conn.commit()
        print(f"  → Inseridos: {i + len(batch)}/{len(vectors)}")

    print("✅ Todos os vetores foram inseridos.")


def main():
    # Caminho para o arquivo .fvecs
    file_path = os.getenv("FILE_PATH")

    try:
        # Ler os vetores do arquivo .fvecs
        vectors = read_fvecs(file_path)
        print("read_fvecs ok")        

        # Inserir os vetores no banco de dados
        insert_vectors_to_oracle(vectors)
        print("insert vectors ok")        

    except Exception as e:
        print(f"Erro ao processar os vetores: {e}")            

if __name__ == "__main__":
    main()
