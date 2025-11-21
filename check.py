import numpy as np
import struct
from typing import Tuple
import argparse


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


def read_ivecs(filename: str) -> np.ndarray:
    """Lê arquivo .ivecs e retorna array numpy."""
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


def calculate_distance(vec1: np.ndarray, vec2: np.ndarray, metric: str = 'euclidean') -> float:
    """
    Calcula distância entre dois vetores.
    
    Args:
        vec1: primeiro vetor
        vec2: segundo vetor
        metric: 'euclidean', 'cosine' ou 'dot'
        
    Returns:
        float: distância calculada
    """
    if metric == 'euclidean':
        return np.linalg.norm(vec1 - vec2)
    elif metric == 'cosine':
        return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    elif metric == 'dot':
        return -np.dot(vec1, vec2)  # Negativo para manter ordem crescente
    else:
        raise ValueError(f"Métrica desconhecida: {metric}")


def verify_groundtruth_consistency(base_vectors: np.ndarray, 
                                   query_vectors: np.ndarray,
                                   groundtruth: np.ndarray,
                                   metric: str = 'euclidean',
                                   sample_size: int = 10) -> dict:
    """
    Verifica consistência entre base, queries e ground truth.
    
    Args:
        base_vectors: vetores da base de dados
        query_vectors: vetores de consulta
        groundtruth: ground truth (IDs dos vizinhos mais próximos)
        metric: métrica de distância usada
        sample_size: número de queries a testar
        
    Returns:
        dict: estatísticas de consistência
    """
    print(f"\n{'='*70}")
    print(f"VERIFICAÇÃO DE CONSISTÊNCIA DO DATASET SIFT")
    print(f"{'='*70}\n")
    
    # Informações básicas
    print(f"📊 Informações do Dataset:")
    print(f"   Base vectors:       {base_vectors.shape} ({base_vectors.dtype})")
    print(f"   Query vectors:      {query_vectors.shape} ({query_vectors.dtype})")
    print(f"   Ground truth:       {groundtruth.shape} ({groundtruth.dtype})")
    print(f"   Métrica:            {metric}")
    print(f"\n{'='*70}\n")
    
    # Verifica dimensões
    assert base_vectors.shape[1] == query_vectors.shape[1], \
        "Base e queries têm dimensões diferentes!"
    
    num_queries = min(len(query_vectors), sample_size)
    k_neighbors = groundtruth.shape[1]
    
    inconsistencies = []
    id_issues = []
    
    print(f"🔍 Testando {num_queries} queries com top-{k_neighbors} vizinhos...\n")
    
    for query_idx in range(num_queries):
        query_vec = query_vectors[query_idx]
        gt_ids = groundtruth[query_idx]
        
        # Verifica se IDs estão dentro do range válido
        max_id = len(base_vectors) - 1
        invalid_ids = gt_ids[gt_ids > max_id]
        if len(invalid_ids) > 0:
            id_issues.append({
                'query_idx': query_idx,
                'invalid_ids': invalid_ids.tolist(),
                'max_valid_id': max_id
            })
            print(f"⚠️  Query {query_idx}: IDs inválidos encontrados: {invalid_ids[:5]}... (max válido: {max_id})")
            continue
        
        # Calcula distâncias reais para os IDs do ground truth
        gt_distances = []
        for neighbor_id in gt_ids[:10]:  # Testa primeiros 10 vizinhos
            base_vec = base_vectors[neighbor_id]
            dist = calculate_distance(query_vec, base_vec, metric)
            gt_distances.append(dist)
        
        # Calcula distâncias para uma amostra aleatória da base
        sample_indices = np.random.choice(len(base_vectors), size=min(1000, len(base_vectors)), replace=False)
        sample_distances = []
        for idx in sample_indices:
            base_vec = base_vectors[idx]
            dist = calculate_distance(query_vec, base_vec, metric)
            sample_distances.append((idx, dist))
        
        # Ordena amostra por distância
        sample_distances.sort(key=lambda x: x[1])
        
        # Verifica se ground truth está consistente
        gt_min_dist = min(gt_distances)
        sample_min_dist = sample_distances[0][1]
        
        if sample_min_dist < gt_min_dist * 0.95:  # Tolerância de 5%
            inconsistencies.append({
                'query_idx': query_idx,
                'gt_min_distance': gt_min_dist,
                'sample_min_distance': sample_min_dist,
                'gt_first_id': gt_ids[0],
                'sample_first_id': sample_distances[0][0]
            })
        
        # Mostra progresso
        if (query_idx + 1) % 10 == 0 or query_idx == 0:
            print(f"   Query {query_idx:3d}: GT[0]={gt_ids[0]:5d} (dist={gt_distances[0]:.4f}), "
                  f"Sample[0]={sample_distances[0][0]:5d} (dist={sample_distances[0][1]:.4f})")
    
    print(f"\n{'='*70}\n")
    
    # Relatório final
    print(f"📈 RESULTADOS DA VERIFICAÇÃO:\n")
    
    if len(id_issues) > 0:
        print(f"❌ PROBLEMA CRÍTICO: IDs fora do range encontrados!")
        print(f"   {len(id_issues)} queries têm IDs inválidos no ground truth")
        print(f"   Range válido de IDs: 0 a {len(base_vectors) - 1}")
        print(f"   Possível causa: Ground truth e base não correspondem\n")
        
        print(f"   Exemplos de IDs inválidos:")
        for issue in id_issues[:3]:
            print(f"   - Query {issue['query_idx']}: IDs {issue['invalid_ids'][:5]} (max válido: {issue['max_valid_id']})")
    else:
        print(f"✅ Todos os IDs do ground truth estão no range válido (0 a {len(base_vectors) - 1})")
    
    print(f"\n")
    
    if len(inconsistencies) > 0:
        print(f"⚠️  {len(inconsistencies)} inconsistências encontradas:")
        print(f"   O ground truth pode estar usando métrica diferente ou dataset diferente\n")
        
        print(f"   Exemplos de inconsistências:")
        for inc in inconsistencies[:3]:
            print(f"   - Query {inc['query_idx']}:")
            print(f"     GT: ID={inc['gt_first_id']}, dist={inc['gt_min_distance']:.6f}")
            print(f"     Encontrado: ID={inc['sample_first_id']}, dist={inc['sample_min_distance']:.6f}")
    else:
        print(f"✅ Ground truth consistente com a base (métrica: {metric})")
    
    print(f"\n{'='*70}\n")
    
    # Recomendações
    print(f"💡 RECOMENDAÇÕES:\n")
    
    if len(id_issues) > 0:
        print(f"1. ⚠️  CRÍTICO: Os IDs no ground truth não correspondem à base!")
        print(f"   - Verifique se está usando os arquivos corretos")
        print(f"   - siftsmall: base tem 10.000 vetores (IDs: 0-9999)")
        print(f"   - sift10k: base tem 10.000 vetores (IDs: 0-9999)")
        print(f"   - sift1M: base tem 1.000.000 vetores (IDs: 0-999999)")
        print(f"\n2. ⚠️  No Oracle, os IDs devem corresponder:")
        print(f"   - Se ground truth usa IDs 1-10000, insira com ID+1 no Oracle")
        print(f"   - Se ground truth usa IDs 0-9999, mantenha como está")
    elif len(inconsistencies) > 0:
        print(f"1. O ground truth pode estar usando métrica diferente")
        print(f"   - Tente: euclidean, cosine, dot")
        print(f"   - SIFT geralmente usa: Euclidean (L2)")
        print(f"\n2. No Oracle AI Vector Search, use a métrica correspondente:")
        print(f"   - EUCLIDEAN para distância L2")
        print(f"   - COSINE para similaridade de cosseno")
        print(f"   - DOT para produto escalar")
    else:
        print(f"✅ Dataset está consistente!")
        print(f"\n⚠️  Se recall ainda está baixo, verifique:")
        print(f"1. IDs no Oracle começam em 1 ou 0? (ground truth geralmente usa 0)")
        print(f"2. Métrica usada na query Oracle corresponde ao ground truth")
        print(f"3. Todos os vetores foram inseridos corretamente")
    
    print(f"\n{'='*70}\n")
    
    return {
        'total_queries_tested': num_queries,
        'id_issues': len(id_issues),
        'inconsistencies': len(inconsistencies),
        'consistency_rate': 1 - (len(inconsistencies) / num_queries) if num_queries > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description='Verifica consistência do dataset SIFT')
    parser.add_argument('--base', type=str, default='dataset/siftsmall_base.fvecs',
                       help='Arquivo base (default: dataset/siftsmall_base.fvecs)')
    parser.add_argument('--query', type=str, default='dataset/siftsmall_query.fvecs',
                       help='Arquivo de queries (default: dataset/siftsmall_query.fvecs)')
    parser.add_argument('--groundtruth', type=str, default='dataset/siftsmall_groundtruth.ivecs',
                       help='Arquivo ground truth (default: dataset/siftsmall_groundtruth.ivecs)')
    parser.add_argument('--metric', type=str, default='euclidean',
                       choices=['euclidean', 'cosine', 'dot'],
                       help='Métrica de distância (default: euclidean)')
    parser.add_argument('--sample', type=int, default=10,
                       help='Número de queries a testar (default: 10)')
    
    args = parser.parse_args()
    
    try:
        # Carrega arquivos
        print(f"Carregando arquivos...")
        base_vectors = read_fvecs(args.base)
        query_vectors = read_fvecs(args.query)
        groundtruth = read_ivecs(args.groundtruth)
        
        # Verifica consistência
        stats = verify_groundtruth_consistency(
            base_vectors=base_vectors,
            query_vectors=query_vectors,
            groundtruth=groundtruth,
            metric=args.metric,
            sample_size=args.sample
        )
        
    except FileNotFoundError as e:
        print(f"❌ Erro: Arquivo não encontrado - {e}")
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()