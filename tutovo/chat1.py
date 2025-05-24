
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import nltk
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
import nltk
nltk.download('punkt_tab')
  # Remove acentos

# Baixar recursos do NLTK
nltk.download('punkt')#Tokeniza sentenças em palavras.
nltk.download('wordnet')#Permite a lemmatização, reduzindo palavras à sua forma básica.

lemmatizer = WordNetLemmatizer()

sup_idoso = {
    "ligar": "📞 Para ligar:\n1. Toque no ícone do telefone\n2. Toque no campo de pesquisa\n3. Digite o nome do contato\n4. Toque no nome quando aparecer\n5. Toque no ícone verde do telefone",
    "foto": "📸 Para tirar foto:\n1. Toque no icone da câmera\n2. Aponte para o que quer fotografar\n3. Toque no círculo grande embaixo\n4. Toque na galeria para ver a foto",
    "galeria": "🖼️ Para abrir a galeria de fotos:\n1. Procure e toque no ícone 'Galeria' ou 'Fotos' na tela inicial\n2. Navegue pelas fotos deslizando para cima ou para os lados\n3. Toque em uma foto para ampliar",
    "agenda": "📒 Para acessar a agenda de contatos:\n1. Toque no ícone 'Contatos' ou 'Agenda'\n2. Procure o nome do contato na lista ou use a barra de busca\n3. Toque no nome para ver detalhes ou ligar",
    "volume": "🔊 Ajustar volume:\n1. Pressione o botão volume superior na lateral do celular para aumentar\n2. Pressione o botão volume inferior na lateral do celular para diminuir\n3. Mantenha pressionado o botão volume inferior para modo silencioso",
    "wifi": "🌐 Conectar ao Wi-Fi:\n1. Deslize o dedo de cima para baixo na tela\n2. Toque em Wi-Fi\n3. Toque no interruptor para ligar\n4. Escolha sua rede\n5. Digite a senha e toque em Conectar",
    "lanterna": "🔦 Para ligar a lanterna:\n1. Deslize o dedo de cima para baixo na tela\n2. Procure o ícone de lanterna\n3. Toque no ícone para acender ou apagar",
    "desinstalar_app": "🗑️ Para desinstalar um aplicativo:\n1. Toque e segure o ícone do app na tela\n2. Arraste até 'Desinstalar' ou toque em 'Remover'\n3. Confirme a exclusão",
    "youtube": "▶️ Para ver vídeos no YouTube:\n1. Abra o app YouTube\n2. Toque na barra de pesquisa\n3. Digite o nome do vídeo ou canal\n4. Toque no vídeo desejado para assistir",
    "baixar_app": "🛒 Para baixar um aplicativo:\n1. Abra o app 'Play Store' (Android) ou 'App Store' (iPhone)\n2. Toque na barra de pesquisa e digite o nome do aplicativo desejado\n3. Toque no nome do app na lista\n4. Toque em 'Instalar' ou 'Obter'\n5. Aguarde o download terminar e abra o app",
    "ajuste_fonte": "🔠 Para aumentar o tamanho da letra:\n1. Abra 'Configurações'\n2. Toque em 'Visor' ou 'Tela'\n3. Procure 'Tamanho da fonte'\n4. Mova a barra para aumentar o texto",
    "teclado_maior": "⌨️ Para aumentar o teclado:\n1. Abra o app de Configurações\n2. Toque em 'Sistema' ou 'Gerenciamento geral'\n3. Selecione 'Teclado' ou 'Idioma e entrada'\n4. Toque em 'Configurações do teclado'\n5. Procure a opção de 'Altura do teclado' e selecione maior",
    "modo_facil": "🧓 Para ativar o modo fácil (Samsung):\n1. Abra 'Configurações'\n2. Toque em 'Visor'\n3. Toque em 'Modo Fácil'\n4. Ative a opção para deixar os ícones maiores e o celular mais simples",
    "emergencia": "🚨 Para usar o botão de emergência:\n1. Procure um botão vermelho ou de 'SOS' no seu celular (alguns modelos possuem)\n2. Pressione e segure para chamar um contato de emergência\n3. Você pode cadastrar números de familiares para receberem o alerta",
    "configuracoes_basicas": "⚙️ Para configurar o celular:\n1. Abra 'Configurações'\n2. Ajuste o brilho em 'Visor'\n3. Para senha, toque em 'Tela de bloqueio' e escolha uma opção simples\n4. Para idioma, vá em 'Idioma e entrada'\n5. Siga as instruções na tela",
    "google_assistente": "🗣️ Para usar o Google Assistente:\n1. Diga 'Ok Google' ou pressione o botão do microfone\n2. Fale o que deseja, como 'Que horas são?' ou 'Abrir WhatsApp'\n3. Aguarde a resposta do assistente",
    
    # Whastapp
    "mensagem": "💬 Para enviar mensagem:\n1. Toque no ícone do WhatsApp\n2. Toque no ícone de nova conversa (+)\n3. Escolha o contato\n4. Escreva sua mensagem\n5. Toque no ícone de enviar (seta)",
    "mensagem_voz": "🎤 Para enviar mensagem de voz:\n1. Toque no icone do WhatsApp\n2. Selecione o contato\n3. Toque e segure o ícone de microfone ao lado do campo de mensagem\n4. Fale sua mensagem\n5. Solte para enviar",
    "videochamada": "📹 Para fazer uma videochamada no WhatsApp:\n1. Abra o WhatsApp\n2. Toque no contato desejado\n3. Toque no ícone de câmera no topo da tela\n4. Aguarde a pessoa atender para conversar por vídeo",
    "pix": "💸 Para cadastrar ou usar o Pix:\n1. Abra o app do seu banco\n2. Procure a opção 'Pix' no menu\n3. Siga as instruções para cadastrar sua chave (pode ser CPF, telefone, e-mail ou chave aleatória)\n4. Para transferir, escolha 'Transferir via Pix', digite a chave do destinatário, o valor e confirme",
    
    # UBER
    "uber_pedir_corrida": "🚗 Para pedir uma corrida no Uber:\n1. Abra o app Uber\n2. Toque em 'Para onde?'\n3. Digite o endereço de destino\n4. Confirme o local de partida\n5. Escolha o tipo de carro\n6. Toque em 'Confirmar Uber' e aguarde o motorista chegar[4].",
    "uber_forma_pagamento": "💳 Para escolher ou adicionar forma de pagamento no Uber:\n1. Abra o app e toque em 'Conta'\n2. Vá em 'Pagamento'\n3. Toque em 'Adicionar forma de pagamento' para cadastrar cartão, Pix, dinheiro, boleto ou outros\n4. Antes de pedir a corrida, toque na forma de pagamento e escolha a desejada[3].",
    "uber_historico_corridas": "📋 Para consultar o histórico de corridas no Uber:\n1. Abra o app Uber\n2. Toque no menu (três linhas ou sua foto)\n3. Selecione 'Histórico'\n4. Veja todas as corridas feitas, valores e detalhes[7].",
    "uber_versao_idoso": "👵 O Uber está testando uma versão simplificada para idosos, com letras maiores e menos botões. Também é possível pedir para um familiar agendar corridas para você usando o perfil familiar[4].",

    # 99
    "99_pedir_corrida": "🚕 Para pedir uma corrida no 99:\n1. Abra o app 99\n2. Toque em 'Para onde vamos?'\n3. Digite o endereço de destino\n4. Confirme o local de partida\n5. Escolha a categoria (99POP, 99TAXI, etc.)\n6. Toque em 'Confirmar Corrida' e aguarde o motorista chegar[6].",
    "99_forma_pagamento": "💳 Para escolher ou cadastrar forma de pagamento no 99:\n1. Toque em 'Para onde vamos?'\n2. Depois, toque na forma de pagamento (abaixo do valor)\n3. Escolha entre dinheiro, cartão, PayPal, 99Pay ou maquininha\n4. Para adicionar cartão, toque em 'Adicionar cartão', preencha os dados e confirme[5].",
    "99_modalidades": "🚖 O app 99 tem várias modalidades:\n- 99POP: motoristas particulares (mais barato)\n- 99TAXI: táxis comuns\n- 99TOP: táxis de luxo\nEscolha a modalidade na tela de seleção da corrida[6].",

    # MEDSÊNIOR
    "medsenior_acesso": "🏥 Para acesssar informações no app MedSênior:\n1. Abra o app MedSênior\n2. Faça login com seus dados\n3. No menu, escolha 'Consultas', 'Carteirinha', 'Rede credenciada' ou 'Boletos' para acessar o que precisa.",
    "medsenior_consulta": "📅 Para marcar consulta no app MedSênior:\n1. Abra o app\n2. Toque em 'Consultas' ou 'Agendamento'\n3. Escolha o médico, data e horário\n4. Confirme e aguarde a confirmação.",
    "medsenior_carteirinha": "🪪 Para ver sua carteirinha digital MedSênior:\n1. Abra o app\n2. Toque em 'Carteirinha' no menu principal\n3. Mostre a tela na hora da consulta, se necessário.",

    # UNIMED
    "unimed_acesso": "💚 Para acessar informações no app Unimed:\n1. Abra o app Unimed\n2. Faça login com seus dados\n3. No menu, escolha 'Consultas', 'Carteirinha', 'Rede credenciada' ou 'Boletos'.",
    "unimed_consulta": "📅 Para marcar consulta no app Unimed:\n1. Abra o app\n2. Toque em 'Agendamento' ou 'Consultas'\n3. Escolha o médico, data e horário\n4. Confirme e aguarde a confirmação.",
    "unimed_carteirinha": "🪪 Para acessar sua carteirinha digital Unimed:\n1. Abra o app\n2. Toque em 'Carteirinha' no menu principal\n3. Mostre a tela na hora da consulta.",
    "unimed_documentos": "📄 Para enviar documentos pelo app Unimed:\n1. Abra o app\n2. Procure a opção 'Enviar documentos' ou 'Documentos'\n3. Siga as instruções para anexar e enviar."

}

intents = {
    "intents": [
        {"tag": "ligar", 
            "patterns": ["ligar para meu neto", "fazer chamada", "como telefonar", "quero ligar", "fazer ligação"],
            "responses": [sup_idoso["ligar"]]},
        {"tag": "mensagem",
            "patterns": ["enviar mensagem", "mandar recado", "escrever para minha filha", "mensagem no whatsapp", "mandar texto"],
            "responses": [sup_idoso["mensagem"]]},
        {"tag": "mensagem_voz",
            "patterns": ["enviar mensagem de voz", "mandar áudio", "gravar mensagem", "falar no whatsapp"],
            "responses": [sup_idoso["mensagem_voz"]]},
        {"tag": "foto",
            "patterns": ["tirar foto", "fotografar neto", "como usar a câmera", "fazer foto", "abrir câmera"],
            "responses": [sup_idoso["foto"]]},
        {"tag": "galeria",
            "patterns": ["abrir galeria", "ver fotos", "onde estão minhas fotos", "mostrar fotos", "galeria de imagens"],
            "responses": [sup_idoso["galeria"]]},
        {"tag": "agenda",
            "patterns": ["acessar agenda", "ver contatos", "abrir contatos", "procurar telefone", "listar contatos"],
            "responses": [sup_idoso["agenda"]]},
        {"tag": "volume",
            "patterns": ["aumentar volume", "celular muito baixo", "não estou ouvindo", "deixar mais alto", "diminuir volume"],
            "responses": [sup_idoso["volume"]]},
        {"tag": "wifi",
            "patterns": ["conectar internet", "wi-fi não funciona", "como entrar na internet", "ligar wi-fi", "acessar wi-fi"],
            "responses": [sup_idoso["wifi"]]},
        {"tag": "lanterna",
            "patterns": ["ligar lanterna", "preciso de luz", "acender lanterna", "usar lanterna do celular", "onde fica a lanterna"],
            "responses": [sup_idoso["lanterna"]]},
        {"tag": "desinstalar_app",
            "patterns": ["excluir aplicativo", "remover app", "apagar programa", "desinstalar aplicativo", "tirar app do celular"],
            "responses": [sup_idoso["desinstalar_app"]]},
        {"tag": "youtube",
            "patterns": ["ver vídeo", "abrir youtube", "assistir vídeos", "procurar vídeo no youtube", "ver canal no youtube"],
            "responses": [sup_idoso["youtube"]]},
        {"tag": "google_assistente",
            "patterns": ["usar assistente", "ativar google", "falar com celular", "como usar assistente", "ok google"],
            "responses": [sup_idoso["google_assistente"]]},
        {"tag": "baixar_app",
            "patterns": ["baixar aplicativo", "instalar app", "como baixar whatsapp", "quero instalar aplicativo", "como colocar app no celular"],
            "responses": [sup_idoso["baixar_app"]]},
        {"tag": "videochamada",
            "patterns": ["fazer videochamada", "ligação de vídeo", "chamar por vídeo", "ver meu neto em vídeo", "chamada de vídeo whatsapp"],
            "responses": [sup_idoso["videochamada"]]},
        {"tag": "pix",
            "patterns": ["cadastrar pix", "como usar pix", "fazer transferência pix", "receber dinheiro pelo pix", "chave pix"],
            "responses": [sup_idoso["pix"]]},
        {"tag": "ajuste_fonte",
            "patterns": ["aumentar letra", "letra pequena", "tamanho da fonte", "deixar texto maior", "dificuldade para ler no celular"],
            "responses": [sup_idoso["ajuste_fonte"]]},
        {"tag": "teclado_maior",
            "patterns": ["teclado pequeno", "aumentar teclado", "teclas pequenas", "dificuldade de digitar"],
            "responses": [sup_idoso["teclado_maior"]]},
        {"tag": "modo_facil",
            "patterns": ["modo fácil", "celular mais simples", "deixar celular fácil", "facilitar uso do celular"],
            "responses": [sup_idoso["modo_facil"]]},
        {"tag": "emergencia",
            "patterns": ["botão de emergência", "chamar socorro", "ajuda rápida", "emergência no celular"],
            "responses": [sup_idoso["emergencia"]]},
        {"tag": "configuracoes_basicas",
            "patterns": ["configurar celular", "ajustar brilho", "colocar senha", "configurar idioma", "primeiros passos no celular"],
            "responses": [sup_idoso["configuracoes_basicas"]]},
        # UBER
        {"tag": "uber_pedir_corrida",
            "patterns": ["como pedir uber", "chamar uber", "usar uber", "corrida no uber", "quero um uber"],
            "responses": [sup_idoso["uber_pedir_corrida"]]},
        {"tag": "uber_forma_pagamento",
            "patterns": ["uber forma de pagamento", "adicionar cartão no uber", "pagar uber com dinheiro", "como pagar uber", "trocar pagamento uber"],
            "responses": [sup_idoso["uber_forma_pagamento"]]},
        {"tag": "uber_historico_corridas",
            "patterns": ["ver corridas uber", "histórico uber", "consultar viagens uber", "corridas antigas uber", "quanto gastei no uber"],
            "responses": [sup_idoso["uber_historico_corridas"]]},
        {"tag": "uber_versao_idoso",
            "patterns": ["uber para idoso", "versão fácil uber", "uber simplificado", "uber para terceira idade", "uber para idosos"],
            "responses": [sup_idoso["uber_versao_idoso"]]},

        # 99
        {"tag": "99_pedir_corrida",
            "patterns": ["como pedir 99", "chamar 99", "usar 99pop", "corrida no 99", "quero um 99"],
            "responses": [sup_idoso["99_pedir_corrida"]]},
        {"tag": "99_forma_pagamento",
            "patterns": ["99 forma de pagamento", "adicionar cartão no 99", "pagar 99 com dinheiro", "como pagar 99", "trocar pagamento 99"],
            "responses": [sup_idoso["99_forma_pagamento"]]},
        {"tag": "99_modalidades",
            "patterns": ["diferença 99pop e 99taxi", "modalidades do 99", "o que é 99pop", "99top", "tipos de corrida 99"],
            "responses": [sup_idoso["99_modalidades"]]},

        # MEDSÊNIOR
        {"tag": "medsenior_acesso",
            "patterns": ["entrar na medsenior","informações medsenior", "app medsenior", " boleto medsenior"],
            "responses": [sup_idoso["medsenior_acesso"]]},
        {"tag": "medsenior_consulta",
            "patterns": ["marcar consulta medsenior", "agendar consulta medsenior", "consultar com o doutor da medsenior","consulta no app medsenior", "como marcar médico medsenior"],
            "responses": [sup_idoso["medsenior_consulta"]]},
        {"tag": "medsenior_carteirinha",
            "patterns": ["carteirinha medsenior", "ver carteirinha medsenior", "carteira digital medsenior", "app medsenior carteirinha"],
            "responses": [sup_idoso["medsenior_carteirinha"]]},

        # UNIMED
        {"tag": "unimed_acesso",
            "patterns": ["acesso na unimed", "informações unimed", "app unimed", "boleto unimed"],
            "responses": [sup_idoso["unimed_acesso"]]},
        {"tag": "unimed_consulta",
            "patterns": ["marcar consulta unimed", "agendar consulta unimed","consultar com o doutor da unimed", "consulta no app unimed", "como marcar médico unimed"],
            "responses": [sup_idoso["unimed_consulta"]]},
        {"tag": "unimed_carteirinha",
            "patterns": ["carteirinha unimed", "ver carteirinha unimed", "carteira digital unimed", "app unimed carteirinha"],
            "responses": [sup_idoso["unimed_carteirinha"]]},
        {"tag": "unimed_documentos",
            "patterns": ["enviar documentos unimed", "mandar documento app unimed", "documentos no app unimed", "como enviar documentos unimed"],
            "responses": [sup_idoso["unimed_documentos"]]}
    ]
}


# Remove acentos e transforma tudo em letras minúsculas.
# Adicionar palavras comuns do universo idoso
ignore_words = ["?", "!", ",", ".", "por", "favor", "quero", "meu", "minha"]

# Aumentar padrões de fala coloquial
def normalize_text(text):
    replacements = {
        "neto": "contato",
        "neta": "contato",
        "filho": "contato",
        "filha": "contato"
    }
    text = unidecode(text.lower())
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text

    
# Processamento dos dados para treino da IA
words = []
classes = []
documents = []
# Tokeniza as frases em palavras. Remove acentos e caracteres indesejados. Armazena palavras, classes e documentos processados.
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        norm_pattern = unidecode(pattern.lower())
        word_list = nltk.word_tokenize(norm_pattern)
        words.extend(word_list)
        documents.append((word_list, unidecode(intent["tag"])))
        if unidecode(intent["tag"]) not in classes:
            classes.append(unidecode(intent["tag"]))
# Lematiza todas as palavras e remove duplicatas
words = sorted(set([lemmatizer.lemmatize(w) for w in words if w not in ignore_words]))
classes = sorted(set(classes))

# Criando dados de treinamento
# Converte frases em vetores binários, associa cada vetor com a classe correta.
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [1 if w in [lemmatizer.lemmatize(word) for word in doc[0]] else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
# Embaralha os dados para evitar viés no aprendizado, separa as entradas (train_x) e saídas (train_y).
random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# AJUSTES PROPORCIONAIS À QUANTIDADE DE INTENTS
num_intents = len(classes)
num_patterns = len(documents)

# Batch size: entre 8 e 20, proporcional ao número de intents
batch_size = min(20, max(8, num_intents // 2))

# Epochs: entre 300 e 600, proporcional ao número de intents
epochs = min(600, max(300, num_intents * 10))

# Ajuste dinâmico da rede neural conforme o tamanho do problema
hidden_1 = min(256, max(64, num_intents * 6))
hidden_2 = min(128, max(32, num_intents * 3))
hidden_3 = min(64, max(16, num_intents))

# Aumentar capacidade de generalização
model = Sequential([
    Dense(hidden_1, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.3),
    Dense(hidden_2, activation='relu'),
    Dropout(0.3),
    Dense(hidden_3, activation='relu'),
    Dense(len(train_y[0]), activation='softmax')
])

# Ajustar hiperparâmetros para melhor aprendizado
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.005),
              metrics=['accuracy'])

# Aumentar épocas de treinamento
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)



# Função para gerar respostas do chatbot
# Converte a entrada do usuário em um vetor de palavras.
# Usa a rede neural para prever a intenção.
# Retorna uma resposta com pelo menos 70% de certeza.
def get_response(user_input):
    user_input = normalize_text(user_input)
    bag = [1 if w in [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(user_input)] else 0 for w in words]
    
    results = model.predict(np.array([bag]), verbose=0)[0]
    max_index = np.argmax(results)
    
    # Aumentar limite de confiança
    if results[max_index] > 0.6:
        return random.choice([r for intent in intents["intents"] 
                            if intent["tag"] == classes[max_index] 
                            for r in intent["responses"]])
    
    # Resposta padrão para baixa confiança
    help_options = "\n".join([f"- {intent['tag'].capitalize()}" 
                            for intent in intents["intents"]])
    return f"Posso ajudar com:\n{help_options}\nPor favor, reformule sua pergunta."

    
# Função para gerar a rotina de estudos
# Distribui horas de estudo para cada disciplina ao longo das semanas.
# Retorna um cronograma personalizado.

# Chatbot interativo
# O chatbot aceita perguntas ou gera um plano de estudos.
def chatbot():
    print("\nTuto aqui, seu axiliador com tecnologia")
    while(True):
        user_input = input("\nVocê: ")
        if user_input != "sair":
            print(get_response(user_input))
        else: break
        
    


#teste: algebra linear, banco de dados, engenharia de software, calculo, inteligencia artificial

if __name__ == "__main__":
    chatbot()
