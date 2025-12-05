import numpy as np
import oracledb
import struct
import time
import threading
from typing import List, Tuple, Dict
from queue import Queue
from dataclasses import dataclass
import argparse
from dotenv import load_dotenv
import os

oracledb.init_oracle_client(lib_dir=r"D:\\instantclient_23_9") # change

# env
load_dotenv() 

@dataclass
class QueryResult:
    """Armazena resultado de uma query individual"""
    query_id: int
    latency: float  # em segundos
    retrieved_ids: List[int]


def read_fvecs(filename: str) -> np.ndarray:
    """
    Lê arquivo .fvecs e retorna array numpy.
    
    Args:
        filename: caminho do arquivo .fvecs
        
    Returns:
        np.ndarray: array 2D com shape (n_vectors, dimension)
    """
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


def read_ivecs(filename: str) -> np.ndarray:
    """
    Lê arquivo .ivecs e retorna array numpy.
    
    Args:
        filename: caminho do arquivo .ivecs
        
    Returns:
        np.ndarray: array 2D com shape (n_vectors, dimension)
    """
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
                
            vec = struct.unpack('i' * dim, vec_bytes)
            vectors.append(vec)
    
    return np.array(vectors, dtype=np.int32)


def connect_database() -> oracledb.Connection:
    """
    Cria e retorna conexão com Oracle Database.
    
    Returns:
        oracledb.Connection: conexão ativa com o banco
    """
    
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    dsn = os.getenv("DB_URL")
    
    connection = oracledb.connect(
        user=username,
        password=password,
        dsn=dsn
    )
    
    return connection


def vector_search(connection: oracledb.Connection, query_vector: np.ndarray, top_k: int) -> Tuple[List[int], float]:
    """
    Executa busca vetorial no Oracle AI Vector Search.
    
    Args:
        connection: conexão com Oracle Database
        query_vector: vetor de consulta
        top_k: número de vizinhos mais próximos a retornar
        
    Returns:
        Tuple[List[int], float]: lista de IDs encontrados e tempo de execução
    """
    cursor = connection.cursor()
    
    # Converte vetor para formato Oracle
    vec_list = query_vector.tolist()
    vec_str = f"[{','.join(map(str, vec_list))}]"
    
    # Mede tempo de execução
    start_time = time.time()
    
    # Query de busca vetorial usando VECTOR_DISTANCE
    cursor.execute(f"""
        SELECT ID
        FROM VECTOR_TABLE
        ORDER BY VECTOR_DISTANCE(EMBEDDING, TO_VECTOR(:vec), EUCLIDEAN)
        FETCH APPROXIMATE FIRST :k ROWS ONLY
    """, {"vec": vec_str, "k": top_k})
    
    results = cursor.fetchall()
    latency = time.time() - start_time
    
    cursor.close()
    
    # Extrai apenas os IDs
    retrieved_ids = [row[0] for row in results]
    
    return retrieved_ids, latency


def worker_thread(thread_id: int, task_queue: Queue, result_list: List[QueryResult], 
                  query_vectors: np.ndarray, top_k: int):
    """
    Thread worker que processa queries da fila.
    
    Args:
        thread_id: ID da thread
        task_queue: fila de tarefas (índices de queries)
        result_list: lista compartilhada para armazenar resultados
        query_vectors: vetores de consulta
        top_k: número de vizinhos a retornar
    """
    # Cada thread tem sua própria conexão
    connection = connect_database()
    
    try:
        while True:
            # Pega próxima tarefa da fila
            query_id = task_queue.get()
            if query_id is None:  # Sinal de parada
                break
            
            # Executa busca vetorial
            query_vector = query_vectors[query_id]
            retrieved_ids, latency = vector_search(connection, query_vector, top_k)
            
            # Armazena resultado
            result = QueryResult(
                query_id=query_id,
                latency=latency,
                retrieved_ids=retrieved_ids
            )
            result_list.append(result)
            
            task_queue.task_done()
            time.sleep(0.001)  # Pequena pausa para evitar sobrecarga
            
    finally:
        connection.close()


def calculate_recall_at_r(results: List[QueryResult], groundtruth: np.ndarray, r: int) -> float:
    """
    Calcula Recall@R para os resultados.
    
    Args:
        results: lista de resultados das queries
        groundtruth: ground truth do benchmark
        r: número de vizinhos a considerar
        
    Returns:
        float: recall@R (0.0 a 1.0)
    """
    total_recall = 0.0
    
    for result in results:
        query_id = result.query_id
        retrieved = set(result.retrieved_ids[:r])
        true_neighbors = set(groundtruth[query_id][:r])
        
        # Calcula interseção
        intersection = len(retrieved.intersection(true_neighbors))
        recall = intersection / r
        total_recall += recall
    
    return total_recall / len(results)


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calcula percentil de uma lista de valores"""
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile)
    return sorted_values[min(index, len(sorted_values) - 1)]


def run_benchmark(query_vectors: np.ndarray, groundtruth: np.ndarray, 
                  num_threads: int = 4, top_k: int = 100, r: int = 100) -> Dict:
    """
    Executa benchmark de busca vetorial com múltiplas threads.
    
    Args:
        query_vectors: vetores de consulta
        groundtruth: ground truth (vizinhos corretos)
        num_threads: número de threads paralelas
        top_k: número de vizinhos a retornar
        r: número de vizinhos para cálculo do recall
        
    Returns:
        Dict: dicionário com estatísticas do benchmark
    """
    num_queries = len(query_vectors)
    print(f"\n{'='*60}")
    print(f"Iniciando benchmark SIFT")
    print(f"{'='*60}")
    print(f"Queries: {num_queries}")
    print(f"Threads: {num_threads}")
    print(f"Top-K: {top_k}")
    print(f"Recall@R: R={r}")
    print(f"{'='*60}\n")
    
    # Cria fila de tarefas
    task_queue = Queue()
    for i in range(num_queries):
        task_queue.put(i)
    
    # Lista compartilhada para resultados (thread-safe com append)
    results = []
    
    # Cria e inicia threads
    threads = []
    start_time = time.time()
    
    for i in range(num_threads):
        thread = threading.Thread(
            target=worker_thread,
            args=(i, task_queue, results, query_vectors, top_k)
        )
        thread.start()
        threads.append(thread)
    
    # Aguarda todas as queries serem processadas
    task_queue.join()
    
    # Envia sinal de parada para threads
    for _ in range(num_threads):
        task_queue.put(None)
    
    # Aguarda threads finalizarem
    for thread in threads:
        thread.join()
    
    total_time = time.time() - start_time
    
    # Calcula estatísticas
    latencies = [r.latency for r in results]
    
    stats = {
        "total_queries": num_queries,
        "total_time": total_time,
        "queries_per_second": num_queries / total_time,
        "avg_latency": np.mean(latencies),
        "min_latency": np.min(latencies),
        "max_latency": np.max(latencies),
        "p90_latency": calculate_percentile(latencies, 0.90),
        "p95_latency": calculate_percentile(latencies, 0.95),
        "p99_latency": calculate_percentile(latencies, 0.99),
        "recall_at_r": calculate_recall_at_r(results, groundtruth, r)
    }
    
    return stats


def print_statistics(stats: Dict):
    """
    Imprime estatísticas do benchmark formatadas.
    
    Args:
        stats: dicionário com estatísticas
    """
    print(f"\n{'='*60}")
    print(f"RESULTADOS DO BENCHMARK")
    print(f"{'='*60}")
    print(f"Total de Queries:        {stats['total_queries']}")
    print(f"Tempo Total:             {stats['total_time']:.2f}s")
    print(f"\n--- THROUGHPUT ---")
    print(f"Queries por Segundo:     {stats['queries_per_second']:.2f} QPS")
    print(f"\n--- LATÊNCIA ---")
    print(f"Latência Média:          {stats['avg_latency']*1000:.2f}ms")
    print(f"Latência Mínima:         {stats['min_latency']*1000:.2f}ms")
    print(f"Latência Máxima:         {stats['max_latency']*1000:.2f}ms")
    print(f"Latência P90:            {stats['p90_latency']*1000:.2f}ms")
    print(f"Latência P95:            {stats['p95_latency']*1000:.2f}ms")
    print(f"Latência P99:            {stats['p99_latency']*1000:.2f}ms")
    print(f"\n--- ACURÁCIA ---")
    print(f"Recall@R:                {stats['recall_at_r']*100:.2f}%")
    print(f"{'='*60}\n")


def main():
    """Função principal que executa o benchmark"""
    
    # Parser de argumentos
    parser = argparse.ArgumentParser(description='SIFT Vector Search Benchmark - Oracle AI')
    parser.add_argument('--threads', type=int, default=4, 
                       help='Número de threads paralelas (default: 4)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Número de vizinhos mais próximos (default: 100)')
    parser.add_argument('--r', type=int, default=3,
                       help='R para cálculo do Recall@R (default: 3)')
    parser.add_argument('--query-file', type=str, default='dataset/sift_query.fvecs',
                       help='Arquivo de queries (default: dataset/sift_query.fvecs)')
    parser.add_argument('--groundtruth-file', type=str, default='dataset/sift_groundtruth.ivecs',
                       help='Arquivo ground truth (default: dataset/sift_groundtruth.ivecs)')
    
    args = parser.parse_args()
    
    try:
        # 1. Carrega queries
        print(f"Carregando queries de '{args.query_file}'...")
        query_vectors = read_fvecs(args.query_file)
        print(f"✓ {len(query_vectors)} queries carregadas (dim={query_vectors.shape[1]})")
        
        # 2. Carrega ground truth
        print(f"Carregando ground truth de '{args.groundtruth_file}'...")
        groundtruth = read_ivecs(args.groundtruth_file)
        print(f"✓ Ground truth carregado ({groundtruth.shape})")
        
        # 3. Executa benchmark
        stats = run_benchmark(
            query_vectors=query_vectors,
            groundtruth=groundtruth,
            num_threads=args.threads,
            top_k=args.top_k,
            r=args.r
        )
        
        # 4. Imprime estatísticas
        print_statistics(stats)
        
    except FileNotFoundError as e:
        print(f"✗ Erro: Arquivo não encontrado - {e}")
    except Exception as e:
        print(f"✗ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()