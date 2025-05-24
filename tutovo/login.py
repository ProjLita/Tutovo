import flet as ft
from chatbot import chatbot_ui

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

