# HealthInsuranceCrossSellPrediction
_________________________________________________________________________
O problema que tentei resolver é o de Cross-Selling de seguro de saúde
baseado em dados do seguro automobilistico.

Os dados podem ser encontrados no endereço 
https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction

Os dados no arquivo train.csv foram usados para o treino e validação do 
modelo e os do arquivo test.csv para gerar um teste para a API.
_________________________________________________________________________
Tratamento Dos Dados

Na coluna 'Gender' os valores foram alterados de 'Male' e'Female' para 1 
e 0 respectivamente.

Na coluna 'Vehicle_Damage' os valores foram alterados de 'Yes' e 'No' 
para 1 e 0 respectivamente.

Na coluna 'Vehicle_Age' os valores foram alterados de '> 2 Years',
'1-2 Year' e '< 1 Year' para 3, 2 e 1 respectivamente.

As colunas 'Region_Code' e 'Policy_Sales_Channel' foram trocadas por 
uma representação One-Hot-Encode de seus valores. 

Estas alterções foram feitas para a melhor interpretação do algoritimo
das informações que os campos representão. Por exemplo o código de 
região e o de Canal de vendas são números mas não existe uma relação 
matematica, são apenas nomes para reprensentar diferentes classes de 
valor.

As colunas 'Annual_Premium', 'Age' e 'Vintage' foram normalizadas com 
normalização Min-Máx para os valores ficarem entre 0 e 1.
______________________________________________________________________
Modelo

O modelo escolhido para ser utilizado na API foi o de Regressão 
Logística da biblioteca Scikit-Learn. Esse modelo é um classificador
binario e foi o que apresentou o melhor resultado das metricas 
utilizadas.

Utilizando a opção de solver="liblinear" a implementação do 
Scikit-Learn 

Também aproveitei a opção class_weight={1:2} na implementação para
tentar compensar o fato que a grande maioria dos exemplos do dataset
de treino tem resposta '0', entã aumentei o peso da classe '1'.  

______________________________________________________________________
Avaliação

O modelo apresentou uma acurácia de 86% e 83% ROC

______________________________________________________________________
API

A API construida foi bem simples, com apenas um ponto de acesso em 
'/predict', um metodo POST que recebe um objeto JSON representando um
ou mais casos de teste para realizar a previsão. 

Para testar utilizei o programa Postman (disponivel em 
https://www.postman.com/ ) para simular o envio do objeto JSON.

Para gerar casos de teste se objeto json, utilizei o arquivo test.csv
no programa csv_to_json.py contido neste zip.
______________________________________________________________________
