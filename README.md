# Streamlit Speech-to-Text App

## Descrição

Este é um aplicativo de transcrição de fala para texto desenvolvido com Streamlit, Whisper, LangChain e Groq. O aplicativo permite que os usuários gravem suas vozes, transcrevam o áudio em texto e processem o texto usando modelos de linguagem avançados.

## Funcionalidades

- **Gravação de Áudio**: Grave sua voz diretamente no navegador.
- **Transcrição de Fala para Texto**: Utilize o Whisper para transcrever o áudio gravado.
- **Processamento de Linguagem Natural**: Use LangChain para realizar análises e manipulações avançadas no texto transcrito.
- **Aceleração de IA**: Utilize o Groq para acelerar o processamento de IA, tornando a transcrição e o processamento de texto mais rápidos e eficientes.

## Tecnologias Utilizadas

- **[Streamlit](https://streamlit.io/)**: Framework para criação de aplicativos web interativos em Python.
- **[Whisper](https://github.com/openai/whisper)**: Modelo de reconhecimento de fala desenvolvido pela OpenAI.
- **[LangChain](https://github.com/langchain/langchain)**: Ferramenta para trabalhar com modelos de linguagem de maneira eficiente.
- **[Groq](https://groq.com/)**: Plataforma de aceleração de IA para otimização de performance.

## Instalação

1. Clone este repositório:

    ```bash
    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd seu-repositorio
    ```

2. Crie um ambiente virtual e ative-o:

    ```bash
    python3 -m venv env
    source env/bin/activate  # No Windows, use `env\Scripts\activate`
    ```

3. Instale as dependências:

    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. Inicie o aplicativo Streamlit:

    ```bash
    streamlit run app.py
    ```

2. No navegador, grave sua voz clicando no botão de gravação.
3. A transcrição do áudio será exibida na tela.
4. Use as funcionalidades adicionais para processar e analisar o texto transcrito.

## Estrutura do Projeto

```plaintext
.
├── app.py                  # Arquivo principal do aplicativo Streamlit
├── requirements.txt        # Arquivo de dependências
├── README.md               # Este arquivo README

```

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.

## Licença
Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
