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

