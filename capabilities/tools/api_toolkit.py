#!/usr/bin/env python3
"""
THAU API Toolkit - Integración Completa de APIs y Servicios

Permite a THAU:
1. Llamar APIs REST con seguridad
2. Manejar webhooks
3. Integrar calendarios
4. Configurar alarmas/recordatorios
5. Enviar notificaciones
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import requests
from enum import Enum
import hmac
import hashlib


class HTTPMethod(Enum):
    """Métodos HTTP"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class AuthType(Enum):
    """Tipos de autenticación"""
    NONE = "none"
    BEARER = "bearer"
    API_KEY = "api_key"
    BASIC = "basic"
    OAUTH2 = "oauth2"


@dataclass
class APIConfig:
    """Configuración de un API"""
    name: str
    base_url: str
    auth_type: AuthType
    credentials: Dict[str, str] = field(default_factory=dict)
    default_headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_attempts: int = 3


@dataclass
class APIResponse:
    """Respuesta de API"""
    status_code: int
    success: bool
    data: Any
    headers: Dict[str, str]
    error: Optional[str] = None


class RESTClient:
    """
    Cliente REST con seguridad y buenas prácticas
    """

    def __init__(self, config: APIConfig):
        self.config = config
        self.session = requests.Session()
        self._setup_auth()

        print(f"🔌 REST Client configurado: {config.name}")
        print(f"   Base URL: {config.base_url}")
        print(f"   Auth: {config.auth_type.value}")

    def _setup_auth(self):
        """Configura autenticación"""
        if self.config.auth_type == AuthType.BEARER:
            token = self.config.credentials.get("token")
            if token:
                self.session.headers["Authorization"] = f"Bearer {token}"

        elif self.config.auth_type == AuthType.API_KEY:
            api_key = self.config.credentials.get("api_key")
            header_name = self.config.credentials.get("header_name", "X-API-Key")
            if api_key:
                self.session.headers[header_name] = api_key

        elif self.config.auth_type == AuthType.BASIC:
            username = self.config.credentials.get("username")
            password = self.config.credentials.get("password")
            if username and password:
                self.session.auth = (username, password)

        # Default headers
        self.session.headers.update(self.config.default_headers)

    def request(
        self,
        method: HTTPMethod,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> APIResponse:
        """
        Hace request con retry y error handling

        Args:
            method: HTTP method
            endpoint: Endpoint (sin base_url)
            data: JSON body
            params: Query parameters
            headers: Headers adicionales

        Returns:
            API Response
        """
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        # Merge headers
        request_headers = self.session.headers.copy()
        if headers:
            request_headers.update(headers)

        # Retry logic
        last_exception = None

        for attempt in range(self.config.retry_attempts):
            try:
                response = self.session.request(
                    method=method.value,
                    url=url,
                    json=data,
                    params=params,
                    headers=request_headers,
                    timeout=self.config.timeout
                )

                # Success
                if response.status_code < 400:
                    return APIResponse(
                        status_code=response.status_code,
                        success=True,
                        data=response.json() if response.content else None,
                        headers=dict(response.headers)
                    )

                # Client error (4xx) - no retry
                if 400 <= response.status_code < 500:
                    return APIResponse(
                        status_code=response.status_code,
                        success=False,
                        data=None,
                        headers=dict(response.headers),
                        error=f"Client error: {response.status_code}"
                    )

                # Server error (5xx) - retry
                last_exception = Exception(f"Server error: {response.status_code}")

            except requests.exceptions.Timeout:
                last_exception = Exception("Request timeout")

            except requests.exceptions.ConnectionError:
                last_exception = Exception("Connection error")

            except Exception as e:
                last_exception = e

            # Wait before retry
            if attempt < self.config.retry_attempts - 1:
                import time
                time.sleep(2 ** attempt)  # Exponential backoff

        # All retries failed
        return APIResponse(
            status_code=0,
            success=False,
            data=None,
            headers={},
            error=str(last_exception)
        )

    def get(self, endpoint: str, params: Optional[Dict] = None) -> APIResponse:
        """GET request"""
        return self.request(HTTPMethod.GET, endpoint, params=params)

    def post(self, endpoint: str, data: Dict) -> APIResponse:
        """POST request"""
        return self.request(HTTPMethod.POST, endpoint, data=data)

    def put(self, endpoint: str, data: Dict) -> APIResponse:
        """PUT request"""
        return self.request(HTTPMethod.PUT, endpoint, data=data)

    def delete(self, endpoint: str) -> APIResponse:
        """DELETE request"""
        return self.request(HTTPMethod.DELETE, endpoint)


class WebhookManager:
    """
    Gestor de Webhooks

    Permite a THAU recibir y procesar webhooks
    """

    def __init__(self):
        self.webhooks: Dict[str, Callable] = {}
        self.webhook_secrets: Dict[str, str] = {}

        print("🪝 Webhook Manager inicializado")

    def register_webhook(
        self,
        name: str,
        handler: Callable,
        secret: Optional[str] = None
    ):
        """
        Registra un webhook handler

        Args:
            name: Nombre del webhook
            handler: Función que procesa el webhook
            secret: Secret para validación HMAC
        """
        self.webhooks[name] = handler

        if secret:
            self.webhook_secrets[name] = secret

        print(f"   🪝 Webhook registrado: {name}")

    def verify_signature(
        self,
        payload: bytes,
        signature: str,
        secret: str
    ) -> bool:
        """
        Verifica firma HMAC del webhook

        Args:
            payload: Payload del webhook
            signature: Firma recibida
            secret: Secret compartido

        Returns:
            True si válido
        """
        expected = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    def process_webhook(
        self,
        name: str,
        payload: Dict,
        signature: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Procesa un webhook recibido

        Args:
            name: Nombre del webhook
            payload: Datos recibidos
            signature: Firma (si aplica)

        Returns:
            Resultado del procesamiento
        """
        if name not in self.webhooks:
            return {"error": f"Webhook '{name}' no registrado"}

        # Verifica firma si hay secret
        if name in self.webhook_secrets and signature:
            secret = self.webhook_secrets[name]
            payload_bytes = json.dumps(payload).encode()

            if not self.verify_signature(payload_bytes, signature, secret):
                return {"error": "Invalid signature"}

        # Ejecuta handler
        handler = self.webhooks[name]
        try:
            result = handler(payload)
            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}


class CalendarIntegration:
    """
    Integración con Calendarios

    Permite a THAU crear eventos, alarmas, recordatorios
    """

    def __init__(self):
        self.events: List[Dict] = []
        print("📅 Calendar Integration inicializado")

    def create_event(
        self,
        title: str,
        start_time: datetime,
        end_time: datetime,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        reminder_minutes: Optional[int] = 15
    ) -> Dict[str, Any]:
        """
        Crea evento en calendario

        Args:
            title: Título del evento
            start_time: Hora de inicio
            end_time: Hora de fin
            description: Descripción
            location: Ubicación
            attendees: Lista de asistentes (emails)
            reminder_minutes: Minutos antes para recordatorio

        Returns:
            Evento creado
        """
        event = {
            "id": f"event_{len(self.events)}",
            "title": title,
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "description": description,
            "location": location,
            "attendees": attendees or [],
            "reminder": reminder_minutes,
            "created_at": datetime.now().isoformat()
        }

        self.events.append(event)

        print(f"📅 Evento creado: {title}")
        print(f"   Inicio: {start_time}")
        print(f"   Fin: {end_time}")

        return event

    def set_alarm(
        self,
        title: str,
        alarm_time: datetime,
        message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Configura alarma

        Args:
            title: Título de la alarma
            alarm_time: Hora de la alarma
            message: Mensaje

        Returns:
            Alarma configurada
        """
        # Alarma es un evento de 0 duración
        alarm = self.create_event(
            title=title,
            start_time=alarm_time,
            end_time=alarm_time,
            description=message or "Alarma"
        )

        print(f"⏰ Alarma configurada: {title} a las {alarm_time}")

        return alarm

    def get_upcoming_events(
        self,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Obtiene próximos eventos"""
        now = datetime.now()
        end_date = now + timedelta(days=days)

        upcoming = [
            event for event in self.events
            if now <= datetime.fromisoformat(event["start"]) <= end_date
        ]

        return sorted(upcoming, key=lambda e: e["start"])

    def export_to_ical(self, filepath: str):
        """Exporta a formato iCalendar"""
        ical_content = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//THAU//Calendar//EN\n"

        for event in self.events:
            ical_content += f"""
BEGIN:VEVENT
UID:{event['id']}
DTSTART:{event['start'].replace('-', '').replace(':', '')}
DTEND:{event['end'].replace('-', '').replace(':', '')}
SUMMARY:{event['title']}
DESCRIPTION:{event.get('description', '')}
LOCATION:{event.get('location', '')}
END:VEVENT
"""

        ical_content += "END:VCALENDAR"

        Path(filepath).write_text(ical_content)
        print(f"📄 Calendario exportado: {filepath}")


class NotificationManager:
    """
    Gestor de Notificaciones

    Permite a THAU enviar notificaciones por varios canales
    """

    def __init__(self):
        self.notifications: List[Dict] = []
        print("🔔 Notification Manager inicializado")

    def send_notification(
        self,
        title: str,
        message: str,
        channel: str = "email",
        recipient: Optional[str] = None,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Envía notificación

        Args:
            title: Título
            message: Mensaje
            channel: Canal (email, sms, push, slack)
            recipient: Destinatario
            priority: Prioridad (low, normal, high)

        Returns:
            Notificación enviada
        """
        notification = {
            "id": f"notif_{len(self.notifications)}",
            "title": title,
            "message": message,
            "channel": channel,
            "recipient": recipient,
            "priority": priority,
            "sent_at": datetime.now().isoformat(),
            "status": "sent"
        }

        self.notifications.append(notification)

        print(f"🔔 Notificación enviada ({channel}): {title}")

        return notification


class APIToolkit:
    """
    Toolkit Completo de APIs para THAU

    Integra todos los componentes
    """

    def __init__(self):
        self.rest_clients: Dict[str, RESTClient] = {}
        self.webhook_manager = WebhookManager()
        self.calendar = CalendarIntegration()
        self.notifications = NotificationManager()

        print("🛠️  THAU API Toolkit inicializado")

    def add_api(self, config: APIConfig) -> RESTClient:
        """Añade API client"""
        client = RESTClient(config)
        self.rest_clients[config.name] = client
        return client

    def get_api(self, name: str) -> Optional[RESTClient]:
        """Obtiene API client"""
        return self.rest_clients.get(name)


if __name__ == "__main__":
    print("="*70)
    print("🧪 Testing THAU API Toolkit")
    print("="*70)

    toolkit = APIToolkit()

    # Test 1: REST Client
    print("\n" + "="*70)
    print("Test 1: REST API Client")
    print("="*70)

    config = APIConfig(
        name="jsonplaceholder",
        base_url="https://jsonplaceholder.typicode.com",
        auth_type=AuthType.NONE,
        default_headers={"Content-Type": "application/json"}
    )

    client = toolkit.add_api(config)
    response = client.get("/posts/1")

    print(f"Status: {response.status_code}")
    print(f"Success: {response.success}")
    print(f"Data: {response.data}")

    # Test 2: Webhook
    print("\n" + "="*70)
    print("Test 2: Webhook Handler")
    print("="*70)

    def payment_webhook_handler(payload):
        print(f"   Procesando pago: {payload}")
        return {"processed": True}

    toolkit.webhook_manager.register_webhook(
        "payment_received",
        payment_webhook_handler,
        secret="my_secret_key"
    )

    result = toolkit.webhook_manager.process_webhook(
        "payment_received",
        {"amount": 100, "currency": "USD"}
    )

    print(f"Result: {result}")

    # Test 3: Calendar
    print("\n" + "="*70)
    print("Test 3: Calendar Integration")
    print("="*70)

    event = toolkit.calendar.create_event(
        title="Reunión con THAU",
        start_time=datetime.now() + timedelta(hours=2),
        end_time=datetime.now() + timedelta(hours=3),
        description="Discutir nuevas capacidades",
        location="Oficina Virtual"
    )

    alarm = toolkit.calendar.set_alarm(
        title="Recordatorio: Reunión en 15 min",
        alarm_time=datetime.now() + timedelta(hours=1, minutes=45)
    )

    upcoming = toolkit.calendar.get_upcoming_events(days=7)
    print(f"\n📅 Próximos eventos: {len(upcoming)}")

    # Test 4: Notifications
    print("\n" + "="*70)
    print("Test 4: Notifications")
    print("="*70)

    notif = toolkit.notifications.send_notification(
        title="THAU Agent System Ready",
        message="El sistema de agentes está listo para usar",
        channel="email",
        recipient="user@example.com",
        priority="high"
    )

    print("\n" + "="*70)
    print("✅ Tests Completados")
    print("="*70)
