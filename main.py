import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from datetime import datetime
import matplotlib.pyplot as plt
import gc

#Remover!
PATH = "recl_tim_jan_2025.csv"#'logs\\recl_tim_20250110.csv'

def load_data(Path:str)->pd.DataFrame:
    df = pd.read_csv(Path, usecols=['tipo_assinante',
                                    'tipo_terminal',
                                    'simcard_4g',
                                    'mobile_4g',
                                    'voz_4g',
                                    'uso_voz',
                                    'dados_4g_5g',
                                    'uso_dados',
                                    'dsc_origem',
                                    'total_protocolos',
                                    'ath', 
                                'tripleta_tecnica',
                                'flg_ressarcimento',
                                'anatel_120_a',
                                'anatel_bloqueio_120_a',
                                'anatel_atendimento_120_a',
                                'anatel_instalacao_120_a', 
                                'anatel_credito_120_a',
                                'anatel_planos_120_a',
                                'anatel_tecnico_120_a',
                                'anatel_cobranca_120_a',
                                'anatel_cancelamento_120_a',
                                'anatel_cadastro_120_a',
                                'anatel_portabilidade_120_a',
                                'anatel_ressarcimento_120_a',
                                'anatel_30_d'])
    df["anatel_30_d"] = (df["anatel_30_d"] >= 1).astype(int) # transforma e de ocorrência binária 0 não 1 sim
    # One Hot encoding para variáveis categóricas ("Tripleta" ignorado por enquanto)
    # final_df = df.copy()
    #So cria novas colunas com mais de 1% de relevancia
    encoder = OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=0.01, sparse_output=False)

    for col in df.select_dtypes(['object']).columns:
        # Fit and transform the col
        encoded_features = encoder.fit_transform(df[[col]])

        # Create a DataFrame from the encoded features
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out([col]))

        # Concatenate with the original DataFrame (excluding the original col)
        df = pd.concat([df.drop(col, axis=1), encoded_df], axis=1)

    return df.dropna()

def wirte_data(rf, X_test, y_test)->None:
    y_pred = rf.predict(X_test)
    #y_pred_prob = rf.predict_proba(X_test)[:, 1]
    print(y_pred.shape)
    y_prob = rf.predict_proba(X_test)[:, 1]
    with open("Results.txt","r+", encoding='utf-8') as f:
        now_local = datetime.now()
        now_local_str = now_local.strftime("%Y-%m-%d %H:%M:%S")
        f.write('Acurácia:\n')
        f.write(f"{accuracy_score(y_test, y_pred)}")
        f.write("\nROC-AUC:")
        f.write(f'{roc_auc_score(y_test, y_prob)}')
        f.write('\nMatriz de confusão:\n')
        f.writelines([str(i) for i in confusion_matrix(y_test, y_pred)])
        f.write('\nRelatório de classificação:\n')
        f.writelines([str(i) for i in classification_report(y_test, y_pred)])
        f.write('\n'+now_local_str+'\n')
        
def get_shap_top_features(shap_values, X, top_k=20, per_class=False):
    # Normaliza formatos para arr shape = (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        arr = np.stack(shap_values, axis=-1)   # (n_samples, n_features, n_classes)
    else:
        arr = np.array(shap_values)
        if arr.ndim == 2:
            # (n_samples, n_features) -> adicionar uma "classe" única
            arr = arr[:, :, np.newaxis]
        elif arr.ndim == 3:
            # formato (n_classes, n_samples, n_features)
            # detecta e corrige caso seja esse o caso
            n0, n1, n2 = arr.shape
            if n0 == X.shape[1] and n1 == X.shape[0]:
                # improvável, mas checagem defensiva; reorganiza para (n_samples,n_features,n_classes)
                arr = np.transpose(arr, (1,2,0))
            # assume agora arr está ok: (n_samples, n_features, n_classes)

    n_samples, n_features, n_classes = arr.shape
    feat_names = list(X.columns)

    importance_global = np.mean(arr, axis=(0,2))  # shape -> (n_features,)

    df_global = pd.DataFrame({
        "feature": feat_names,
        "importance": importance_global
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    if per_class:
        importance_per_class = np.mean(arr, axis=0)  # (n_features, n_classes)
        df_per_class = pd.DataFrame(
            importance_per_class,
            index=feat_names,
            columns=[f"class_{i}" for i in range(n_classes)]
        )
    else:
        df_per_class = None

    top_features = df_global.head(top_k)["feature"].tolist()
    return df_global, df_per_class, top_features

def main():
    df = load_data(PATH)
    # Separar features (X) e target (y)
    X = df.drop('anatel_30_d', axis=1)
    y = df['anatel_30_d']
    
    del df
    lixo = gc.collect()
    print(f'garbage_colector: {lixo}')
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}",f'\ntypes: {type(X)}|{type(X_train)}|{type(y_train)}')
    #print(f'X_train info:\n {X_train.info()}')
    #Remove df desnecessário
    del X, y
    lixo = gc.collect()
    print(f'garbage_colector: {lixo}')
    
    # Treinar o modelo
    print("Treinando o modelo RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    wirte_data(rf, X_test, y_test)
    
    print("Calculando valores SHAP...")
    # Separa uma amostra (Isso reduz o tempo de processamento mantem uma boa fidelidade)
    X_sample = X_train.sample(500, random_state=42)
    print(X_sample.info())
    
    #Shap
    print("Executando shap")
    explainer = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_sample)
    
    #Shap result    
    df_global, df_per_class, top_features = get_shap_top_features(shap_values, X_sample, top_k=20, per_class=False)

    print("Top features (SHAP):")
    print(top_features)

    # Salvar em CSV se quiser
    df_global.to_csv("shap_feature_importance_global.csv", index=False)
    if df_per_class is not None:
        df_per_class.to_csv("shap_feature_importance_per_class.csv")
        
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    
    # Random forest com colunas do shap
    del X_train, X_test, y_train, y_test
    lixo = gc.collect()
    print(f'garbage_colector: {lixo}')
    
    df = load_data(PATH)
    #Seleciona as colunas do shap
    df = df[top_features]
    
    # Separar features (X) e target (y)
    X = df.drop('anatel_30_d', axis=1)
    y = df['anatel_30_d']
    
    del df
    lixo = gc.collect()
    print(f'garbage_colector: {lixo}')
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}",f'\ntypes: {type(X)}|{type(X_train)}|{type(y_train)}')
    #print(f'X_train info:\n {X_train.info()}')
    #Remove df desnecessário
    del X, y
    lixo = gc.collect()
    print(f'garbage_colector: {lixo}')
    
    # Treinar o modelo
    print("Treinando o modelo RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    wirte_data(rf, X_test, y_test)

if __name__ == "__main__":
    main()