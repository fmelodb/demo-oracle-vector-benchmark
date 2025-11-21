import numpy as np
import oracledb
import struct
from typing import Optional
import argparse
from dotenv import load_dotenv
import os

oracledb.init_oracle_client(lib_dir=r"D:\\instantclient_23_9") # change

# env
load_dotenv() 


def read_fvecs(filename: str) -> np.ndarray:
    """Lê arquivo .fvecs e retorna array numpy."""
    vectors = []
    
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
                
            dim = struct.unpack('i', dim_bytes)[0]
            vec_bytes = f.read(dim * 4)
            if len(vec_bytes) != dim * 4:
                break
                
            vec = struct.unpack('f' * dim, vec_bytes)
            vectors.append(vec)
    
    return np.array(vectors, dtype=np.float32)


def connect_database() -> oracledb.Connection:
    """Cria e retorna conexão com Oracle Database."""
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    dsn = os.getenv("DB_URL")
    
    connection = oracledb.connect(
        user=username,
        password=password,
        dsn=dsn
    )
    
    return connection


def create_vector_table(connection: oracledb.Connection, dimension: int = 128, drop_if_exists: bool = True) -> None:
    """
    Cria tabela VECTOR_TABLE com estrutura otimizada.
    
    Args:
        connection: conexão com Oracle Database
        dimension: dimensionalidade dos vetores (default: 128 para SIFT)
        drop_if_exists: se True, recria a tabela
    """
    cursor = connection.cursor()
    
    try:
        if drop_if_exists:
            print("Removendo tabela existente (se houver)...")
            cursor.execute("""
                BEGIN
                    EXECUTE IMMEDIATE 'DROP TABLE IF EXISTS VECTOR_TABLE';
                EXCEPTION
                    WHEN OTHERS THEN
                        IF SQLCODE != -942 THEN
                            RAISE;
                        END IF;
                END;
            """)
        
        print(f"Criando tabela VECTOR_TABLE (dimensão={dimension})...")
        
        # Cria tabela com coluna VECTOR especificando dimensão
        cursor.execute(f"""
            CREATE TABLE VECTOR_TABLE (
                ID INT PRIMARY KEY,
                EMBEDDING VECTOR({dimension}, FLOAT32)
            )
        """)
        
        connection.commit()
        print("✅ Tabela VECTOR_TABLE criada com sucesso\n")
        
    except Exception as e:
        print(f"⚠️  Aviso ao criar tabela: {e}\n")
    finally:
        cursor.close()


def insert_vectors_correct(connection: oracledb.Connection, 
                          vectors: np.ndarray, 
                          start_id: int = 0,
                          batch_size: int = 1000) -> None:
    """
    Insere vetores no Oracle com IDs corretos para o ground truth.
    
    IMPORTANTE: Ground truth SIFT usa IDs começando em 0 (zero-based).
    Portanto, o primeiro vetor deve ter ID=0, segundo ID=1, etc.
    
    Args:
        connection: conexão com Oracle Database
        vectors: array numpy com vetores
        start_id: ID inicial (default: 0 para compatibilidade com ground truth)
        batch_size: tamanho do lote para inserção
    """
    cursor = connection.cursor()
    
    try:
        total_vectors = len(vectors)
        print(f"{'='*70}")
        print(f"INSERINDO VETORES NO ORACLE AI VECTOR SEARCH")
        print(f"{'='*70}")
        print(f"Total de vetores:     {total_vectors}")
        print(f"Dimensão:             {vectors.shape[1]}")
        print(f"ID inicial:           {start_id}")
        print(f"ID final:             {start_id + total_vectors - 1}")
        print(f"Tamanho do batch:     {batch_size}")
        print(f"{'='*70}\n")
        
        # CRÍTICO: IDs devem corresponder ao índice no ground truth
        # Ground truth[0] aponta para vizinhos usando IDs 0-based
        
        for i in range(0, total_vectors, batch_size):
            batch_end = min(i + batch_size, total_vectors)
            batch = vectors[i:batch_end]
            
            # Prepara dados do lote
            data = []
            for idx, vec in enumerate(batch):
                # ID = start_id + índice no array
                # Para ground truth SIFT: start_id deve ser 0
                vector_id = start_id + i + idx
                
                # Converte vetor para string JSON array
                vec_list = vec.tolist()
                vec_str = f"[{','.join(map(str, vec_list))}]"
                
                data.append((vector_id, vec_str))
            
            # Inserção em bulk
            cursor.executemany(
                """
                INSERT INTO VECTOR_TABLE (ID, EMBEDDING)
                VALUES (:1, TO_VECTOR(:2, *, FLOAT32))
                """,
                data
            )
            
            connection.commit()
            
            # Mostra progresso
            progress = (batch_end / total_vectors) * 100
            print(f"Progresso: {batch_end:6d}/{total_vectors} ({progress:5.1f}%) - "
                  f"IDs {start_id + i} a {start_id + batch_end - 1}")
        
        print(f"\n{'='*70}")
        print(f"✅ INSERÇÃO CONCLUÍDA COM SUCESSO!")
        print(f"{'='*70}\n")
        
        # Verifica inserção
        cursor.execute("SELECT COUNT(*), MIN(ID), MAX(ID) FROM VECTOR_TABLE")
        count, min_id, max_id = cursor.fetchone()
        
        print(f"📊 VERIFICAÇÃO:")
        print(f"   Total de registros: {count}")
        print(f"   ID mínimo:          {min_id}")
        print(f"   ID máximo:          {max_id}")
        print(f"   Range esperado:     {start_id} a {start_id + total_vectors - 1}")
        
        if count == total_vectors and min_id == start_id and max_id == start_id + total_vectors - 1:
            print(f"   Status:             ✅ CORRETO\n")
        else:
            print(f"   Status:             ⚠️  INCONSISTENTE\n")
        
    except Exception as e:
        connection.rollback()
        print(f"\n❌ Erro durante inserção: {e}")
        raise
    finally:
        cursor.close()


def create_vector_index(connection: oracledb.Connection, 
                       index_type: str = "HNSW",
                       distance_metric: str = "EUCLIDEAN") -> None:
    """
    Cria índice vetorial para acelerar buscas.
    
    Args:
        connection: conexão com Oracle Database
        index_type: tipo de índice ('IVF' ou 'HNSW')
        distance_metric: métrica de distância ('EUCLIDEAN', 'COSINE', 'DOT')
    """
    cursor = connection.cursor()
    
    try:
        print(f"\n{'='*70}")
        print(f"CRIANDO ÍNDICE VETORIAL")
        print(f"{'='*70}")
        print(f"Tipo:                 {index_type}")
        print(f"Métrica:              {distance_metric}")
        print(f"{'='*70}\n")
        
        # Remove índice anterior se existir
        cursor.execute("""
            BEGIN
                EXECUTE IMMEDIATE 'DROP INDEX IF EXISTS VECTOR_IDX';
            EXCEPTION
                WHEN OTHERS THEN
                    IF SQLCODE != -1418 THEN
                        RAISE;
                    END IF;
            END;
        """)
        
        # Cria índice vetorial
        if index_type.upper() == "HNSW":
            cursor.execute(f"""
                CREATE VECTOR INDEX VECTOR_IDX ON VECTOR_TABLE(EMBEDDING)
                ORGANIZATION INMEMORY NEIGHBOR GRAPH
                DISTANCE {distance_metric}
                WITH TARGET ACCURACY 95
            """)
        elif index_type.upper() == "IVF":
            cursor.execute(f"""
                CREATE VECTOR INDEX VECTOR_IDX ON VECTOR_TABLE(EMBEDDING)
                ORGANIZATION NEIGHBOR PARTITIONS
                DISTANCE {distance_metric}
                WITH TARGET ACCURACY 95
            """)
        
        connection.commit()
        print("✅ Índice criado com sucesso!\n")
        
    except Exception as e:
        print(f"⚠️  Aviso ao criar índice: {e}\n")
    finally:
        cursor.close()


def verify_insertion(connection: oracledb.Connection, vectors: np.ndarray, sample_size: int = 5) -> None:
    """
    Verifica se os vetores foram inseridos corretamente.
    
    Args:
        connection: conexão com Oracle Database
        vectors: array original de vetores
        sample_size: número de vetores a verificar
    """
    cursor = connection.cursor()
    
    print(f"{'='*70}")
    print(f"VERIFICANDO INSERÇÃO")
    print(f"{'='*70}\n")
    
    try:
        indices = np.random.choice(len(vectors), size=min(sample_size, len(vectors)), replace=False)
        
        all_match = True
        for idx in indices:
            # Busca vetor no Oracle
            cursor.execute("""
                SELECT EMBEDDING FROM VECTOR_TABLE WHERE ID = :id
            """, {"id": int(idx)})
            
            result = cursor.fetchone()
            if result is None:
                print(f"❌ ID {idx}: NÃO ENCONTRADO no banco")
                all_match = False
                continue
            
            # Oracle retorna como string, converte para array
            oracle_vec_str = result[0]
            # Remove colchetes e converte para lista de floats
            oracle_vec = np.array([float(x) for x in oracle_vec_str.strip('[]').split(',')], dtype=np.float32)
            original_vec = vectors[idx]
            
            # Compara vetores
            difference = np.max(np.abs(oracle_vec - original_vec))
            
            if difference < 1e-5:
                print(f"✅ ID {idx}: Vetor correto (diff={difference:.2e})")
            else:
                print(f"❌ ID {idx}: Vetor DIFERENTE (diff={difference:.2e})")
                all_match = False
        
        print(f"\n{'='*70}")
        if all_match:
            print(f"✅ TODOS OS VETORES VERIFICADOS ESTÃO CORRETOS")
        else:
            print(f"⚠️  ALGUNS VETORES ESTÃO INCORRETOS")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"❌ Erro durante verificação: {e}\n")
    finally:
        cursor.close()


def main():
    parser = argparse.ArgumentParser(description='Insere vetores SIFT no Oracle corretamente')
    parser.add_argument('--file', type=str, default='dataset/siftsmall_base.fvecs',
                       help='Arquivo de vetores (default: dataset/siftsmall_base.fvecs)')
    parser.add_argument('--start-id', type=int, default=0,
                       help='ID inicial (default: 0 para compatibilidade com ground truth)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Tamanho do batch (default: 1000)')
    parser.add_argument('--create-index', action='store_true',
                       help='Criar índice vetorial após inserção')
    parser.add_argument('--index-type', type=str, default='HNSW',
                       choices=['IVF', 'HNSW'],
                       help='Tipo de índice (default: HNSW)')
    parser.add_argument('--metric', type=str, default='EUCLIDEAN',
                       choices=['EUCLIDEAN', 'COSINE', 'DOT'],
                       help='Métrica de distância (default: EUCLIDEAN)')
    parser.add_argument('--verify', action='store_true',
                       help='Verificar inserção após conclusão')
    
    args = parser.parse_args()
    
    try:
        # 1. Carrega vetores
        print(f"Carregando vetores de '{args.file}'...")
        vectors = read_fvecs(args.file)
        print(f"✅ {len(vectors)} vetores carregados (dimensão={vectors.shape[1]})\n")
        
        # 2. Conecta ao banco
        print("Conectando ao Oracle Database...")
        conn = connect_database()
        print("✅ Conexão estabelecida\n")
        
        # 3. Cria tabela
        create_vector_table(conn, dimension=vectors.shape[1], drop_if_exists=True)
        
        # 4. Insere vetores
        insert_vectors_correct(
            connection=conn,
            vectors=vectors,
            start_id=args.start_id,
            batch_size=args.batch_size
        )
        
        # 5. Verifica inserção (opcional)
        if args.verify:
            verify_insertion(conn, vectors, sample_size=5)
        
        # 6. Cria índice (opcional)
        if args.create_index:
            create_vector_index(conn, index_type=args.index_type, distance_metric=args.metric)
        
        # 7. Dicas finais
        print(f"💡 PRÓXIMOS PASSOS:\n")
        print(f"1. Execute o verificador de consistência:")
        print(f"   python verify_consistency.py --metric {args.metric.lower()}")
        print(f"\n2. Execute o benchmark:")
        print(f"   python benchmark.py --threads 4 --top-k 100 --r 100")
        print(f"\n3. Certifique-se de usar a métrica correta nas queries:")
        print(f"   VECTOR_DISTANCE(EMBEDDING, query_vec, {args.metric})")
        
        conn.close()
        print(f"\n✅ Processo concluído com sucesso!")
        
    except FileNotFoundError as e:
        print(f"❌ Erro: Arquivo não encontrado - {e}")
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()