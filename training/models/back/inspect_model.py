import pickle

MODEL_PATH = "ppo_getup_back_mjx_20260810_230406.pkl"

with open(MODEL_PATH, "rb") as f:
    tupla_dados = pickle.load(f)

print(f"O arquivo contém uma Tupla com {len(tupla_dados)} itens dentro dela.")

for i, item in enumerate(tupla_dados):
    print(f"\n=============================")
    print(f" ITEM {i} DA TUPLA")
    print(f"=============================")
    tipo_item = type(item)
    print(f"Tipo: {tipo_item}")
    
    # Se for um Dicionário normal do Python
    if isinstance(item, dict):
        print(f"É um dicionário com {len(item)} chaves.")
        print(f"Chaves: {list(item.keys())[:10]}")
        
    # Se for um Dicionário do JAX/Flax
    elif tipo_item.__name__ == 'FrozenDict':
        print(f"É um FrozenDict do JAX/Flax.")
        print(f"Chaves: {list(item.keys())}")
        
    # Se for um array ou outra coisa, mostra só um pedacinho
    else:
        print(f"Conteúdo resumido: {str(item)[:200]}")