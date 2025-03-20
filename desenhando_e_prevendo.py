import joblib
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

# Carregar o modelo treinado
modelo = joblib.load("svm_model.pkl")

# Configuração da interface gráfica (Tkinter)
janela = tk.Tk()
janela.title("Desenhe um número")

# Criar um canvas para desenhar com borda destacada
canvas_tamanho = 280
canvas = tk.Canvas(janela, width=canvas_tamanho, height=canvas_tamanho, bg="white")
canvas.pack()

# Adicionar bordas destacadas ao redor do canvas (indicando onde desenhar)
canvas.create_rectangle(10, 10, canvas_tamanho - 10, canvas_tamanho - 10, outline="black", width=3)
canvas.create_text(canvas_tamanho / 2, canvas_tamanho / 2, text="Desenhe aqui", font=("Arial", 16), fill="blue")

# Criar uma imagem para desenhar nela
imagem = Image.new("L", (canvas_tamanho, canvas_tamanho), "white")
draw = ImageDraw.Draw(imagem)

ultimo_x, ultimo_y = None, None

# Função para desenhar no canvas
def desenhar(event):
    global ultimo_x, ultimo_y
    x, y = event.x, event.y
    if ultimo_x and ultimo_y:
        # Criar linha no canvas
        canvas.create_line((ultimo_x, ultimo_y, x, y), fill="black", width=10, capstyle=tk.ROUND)
        # Desenhar linha na imagem (back-end)
        draw.line((ultimo_x, ultimo_y, x, y), fill="black", width=10)
    ultimo_x, ultimo_y = x, y

# Resetar coordenadas ao soltar o botão
def resetar(event):
    global ultimo_x, ultimo_y
    ultimo_x, ultimo_y = None, None

# Função para limpar o canvas
def limpar():
    canvas.delete("all")
    # Adicionar novamente a borda e a mensagem
    canvas.create_rectangle(10, 10, canvas_tamanho - 10, canvas_tamanho - 10, outline="black", width=3)
    canvas.create_text(canvas_tamanho / 2, canvas_tamanho / 2, text="Desenhe aqui", font=("Arial", 16), fill="blue")
    draw.rectangle((0, 0, canvas_tamanho, canvas_tamanho), fill="white")

# Função para verificar se a imagem está vazia
def imagem_vazia():
    # Verificar se todos os pixels da imagem são brancos (sem desenho)
    img_array = np.array(imagem)
    return np.all(img_array == 255)  # 255 é o valor do branco

# Função para processar a imagem e prever o número
def prever():
    if imagem_vazia():
        feedback_label.config(text="A imagem está vazia! Desenhe algo antes de prever.", fg="red")
        return

    # Redimensionar para 28x28 pixels (igual ao MNIST)
    img_redimensionada = imagem.resize((28, 28))

    # Inverter cores (fundo preto e número branco, como no MNIST)
    img_redimensionada = ImageOps.invert(img_redimensionada)

    # Melhorar contraste e espessura dos traços
    img_redimensionada = img_redimensionada.point(lambda x: 0 if x < 120 else 255)

    # Converter para array numpy
    img_array = np.array(img_redimensionada)

    # Normalizar pixels (0 a 1)
    img_array = img_array / 255.0

    # Transformar em vetor de 784 posições
    img_vetor = img_array.reshape(1, -1)

    # Fazer previsão com o modelo
    previsao = modelo.predict(img_vetor)[0]

    # Exibir resultado
    resultado_label.config(text=f"Modelo previu: {previsao}")

# Funções para as teclas de atalho
def tecla_limpar(event):
    limpar()
    feedback_label.config(text="Tecla 'Z' pressionada: Limpando...", fg="red")

def tecla_prever(event):
    prever()
    feedback_label.config(text="Tecla 'X' pressionada: Prevendo...", fg="blue")

# Função para fechar a janela com a tecla ESC
def tecla_fechar(event):
    janela.quit()

# Vincular teclas de atalho
janela.bind("<z>", tecla_limpar)  # Tecla 'Z' para limpar
janela.bind("<x>", tecla_prever)  # Tecla 'X' para prever
janela.bind("<Escape>", tecla_fechar)  # Tecla 'ESC' para fechar

# Alterar o cursor para simular uma caneta
canvas.config(cursor="pencil")

# Eventos do mouse
canvas.bind("<B1-Motion>", desenhar)
canvas.bind("<ButtonRelease-1>", resetar)

# Botões
btn_limpar = tk.Button(janela, text="Limpar", command=limpar)
btn_limpar.pack()
btn_prever = tk.Button(janela, text="Prever", command=prever)
btn_prever.pack()

# Label para mostrar resultado
resultado_label = tk.Label(janela, text="Desenhe um número e clique em 'Prever'", font=("Arial", 12))
resultado_label.pack()

# Label para mostrar feedback da tecla pressionada
feedback_label = tk.Label(janela, text="Pressione uma tecla (Z para limpar, X para prever)", font=("Arial", 10), fg="black")
feedback_label.pack()

# Iniciar interface gráfica
janela.mainloop()
