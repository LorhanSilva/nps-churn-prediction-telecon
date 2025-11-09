import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime
import matplotlib.pyplot as plt

#Remover!
PATH = "recl_tim_jan_2025.csv"

def load_data(Path:str)->pd.DataFrame:
    df = pd.read_csv(Path)
    # Excluir colunas irrelevantes (IDs, timestamps, etc.)
    df.drop(columns=['msisdn_id', 'data_ref','tripleta','anatel_bloqueio_30_d',
    'anatel_atendimento_30_d',
    'anatel_instalacao_30_d',
    'anatel_credito_30_d',
    'anatel_planos_30_d',
    'anatel_tecnico_30_d',
    'anatel_cobranca_30_d',
    'anatel_cancelamento_30_d',
    'anatel_cadastro_30_d',
    'anatel_portabilidade_30_d',
    'anatel_ressarcimento_30_d'], inplace=True)
    
    # One Hot encoding para variáveis categóricas ("Tripleta" ignorado por enquanto)
    from sklearn.preprocessing import OneHotEncoder

    # final_df = df.copy()
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    for col in df.select_dtypes(['object']).columns:
        # Fit and transform the col
        encoded_features = encoder.fit_transform(df[[col]])

        # Create a DataFrame from the encoded features
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out([col]))

        # Concatenate with the original DataFrame (excluding the original col)
        df = pd.concat([df.drop(col, axis=1), encoded_df], axis=1)

    print(df.info())
    return df

def wirte_data(rf, X_test, y_test)->None:
    y_pred = rf.predict(X_test)
    #y_pred_prob = rf.predict_proba(X_test)[:, 1]
    print(y_pred.shape)

    with open("Results.txt","r+", encoding='utf-8') as f:
        now_local = datetime.now()
        now_local_str = now_local.strftime("%Y-%m-%d %H:%M:%S")
        f.write('Acurácia:\n')
        f.write(f"{accuracy_score(y_test, y_pred)}")
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
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}",f'\ntypes: {type(X)}|{type(X_train)}|{type(y_train)}')
    #print(f'X_train info:\n {X_train.info()}')
    
    # Treinar o modelo
    print("Treinando o modelo RandomForest...")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    wirte_data(rf, X_test, y_test)
    
    print("Calculando valores SHAP...")
    # Separa uma amostra (Isso reduz o tempo de processamento mantem uma boa fidelidade)
    X_sample = X_train.sample(10000, random_state=42)
    print(X_sample.info())
    
    #Shap    
    explainer = shap.TreeExplainer(rf)
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
    plt.close()

if __name__ == "__main__":
    main()