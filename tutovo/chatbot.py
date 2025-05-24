import flet as ft
from chat1 import get_response

def chatbot_ui(page: ft.Page):
    page.title = "Assistente do Idoso"
    page.window_width = 390
    page.window_height = 700
    page.bgcolor = ft.Colors.BLUE_100

    chat_display = ft.Column(scroll="auto", expand=True)
    user_input = ft.TextField(
        label="Digite sua dúvida...",
        autofocus=True,
        expand=True,
        on_submit=lambda e: send_message(e),
        text_size=20,
        height=50,
        color=ft.Colors.WHITE
    )

    def send_message(e):
        pergunta = user_input.value.strip()
        if not pergunta:
            return
        chat_display.controls.append(
            ft.Container(
                content=ft.Text(f"Você: {pergunta}", size=18, color=ft.Colors.WHITE),
                alignment=ft.alignment.center_right,
                padding=8,
                bgcolor=ft.Colors.LIGHT_BLUE_300,
                border_radius=12,
                margin=4
            )
        )
        resposta = get_response(pergunta)  # Substitua pela chamada ao seu modelo
        chat_display.controls.append(
            ft.Container(
                content=ft.Text(f"Tuto: {resposta}", size=18, color=ft.Colors.WHITE),
                alignment=ft.alignment.center_left,
                padding=8,
                bgcolor=ft.Colors.GREEN_300,
                border_radius=12,
                margin=4
            )
        )
        user_input.value = ""
        page.update()

    page.add(
        ft.Container(
            content=ft.Text("Olá! Sou o Tuto, seu ajudante com tecnologia.", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE ),
            padding=10,
            alignment=ft.alignment.center
        ),
        ft.Container(
            content=chat_display,
            expand=True,
            bgcolor=ft.Colors.WHITE,
            border_radius=10,
            padding=10,
            margin=5
        ),
        ft.Row(
            controls=[
                user_input,
                ft.FilledButton("Enviar", icon=ft.Icons.SEND, on_click=send_message)
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER
        )
    )

# Função principal com tela de login
def main(page: ft.Page):
    page.title = "Login - Assistente do Idoso"
    page.window_width = 390
    page.window_height = 700
    page.bgcolor = ft.Colors.BLUE_100

    # Centraliza todos os controles filhos da página
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    username = ft.TextField(label="Usuário", autofocus=True)
    password = ft.TextField(label="Senha", password=True, can_reveal_password=True)
    login_msg = ft.Text("", color=ft.Colors.RED_400)

    def login(e):
        if username.value == "idoso" and password.value == "1234":
            page.clean()  # Limpa a tela
            chatbot_ui(page)  # Chama o chatbot
        else:
            login_msg.value = "Usuário ou senha inválidos."
            page.update()

    login_btn = ft.ElevatedButton("Entrar", on_click=login)

    page.add(
        ft.Container(
            content=ft.Column(
                [
                    ft.Text("Login", size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                    username,
                    password,
                    login_btn,
                    login_msg
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                tight=True,
                spacing=10
            ),
            alignment=ft.alignment.center,
            padding=30,
            bgcolor=ft.Colors.BLUE_300,
            border_radius=16,
            width=340
        )
    )

ft.app(target=main)
