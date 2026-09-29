# Sara para OmniStudio (Rasa 3.6)

Gateway autenticado para Sara. Rasa atiende primero cada mensaje del chat de Sara. Si la intención cae en el fallback marcado `sara_fallback`, la app consulta el LLM de Felo para responder en el idioma del mensaje; las herramientas de Felo siguen ejecutándose solo cuando la app o el usuario las solicita explícitamente. Las funciones multimedia de Gemini permanecen separadas.

## Seguridad y despliegue

- Todas las rutas `/api/` requieren `Authorization: Bearer <Firebase ID token>`; el gateway valida el token y usa el UID verificado.
- `FELO_API_KEY` y `RASA_AUTH_TOKEN` se guardan como secretos de Render, nunca en Android ni en GitHub.
- No se reenvían llamadas a herramientas que un LLM pida ejecutar. Los resultados de PPT y páginas se devuelven como vista previa; no se publican ni comparten automáticamente.
- El servicio de Render Free puede tardar en iniciar Rasa después de una suspensión o un nuevo despliegue.

## Funciones Felo disponibles

- Búsqueda web con fuentes, extracción de una URL y subtítulos existentes de YouTube.
- LLM no transmitido, PPT y landing pages como tareas, mapas mentales y SuperAgent.
- X Search con límites locales y un máximo de cinco resultados por búsqueda.
- LiveDocs: creación, actualización y eliminación; recursos, documentos de texto, archivos y URL; recuperación semántica, rutas y extracción de páginas PPT; README; tareas, comentarios y registros; descarga del archivo original.

Las herramientas de Felo no se ejecutan automáticamente por un mensaje ordinario: requieren una solicitud explícita. Excepción: cuando Rasa marca `sara_fallback`, OmniStudio envía ese mensaje al LLM de Felo para responder en el idioma original; esto puede consumir créditos. Ese fallback no ejecuta herramientas ni aprende automáticamente de las conversaciones. Para mejorar Rasa, los ejemplos de cada idioma se agregan y revisan manualmente antes de volver a entrenar. Una extracción de URL no es una búsqueda web; la búsqueda debe solicitarse por separado.

## LiveDocs: referencias seguras por usuario

Crear un LiveDoc:

- `POST /api/felo/livedocs` con JSON como `{"name":"Notas","description":"..."}`.
- La respuesta incluye un `doc_ref` opaco firmado y asociado al UID autenticado. Guárdalo en almacenamiento privado del usuario dentro de la app.
- En todas las operaciones posteriores envía `X-Felo-LiveDoc-Ref: <doc_ref>`. El ID real de Felo no se acepta desde el cliente ni se devuelve en claro.
- La lista se consulta con `POST /api/felo/livedocs/list` y un cuerpo con `doc_refs` del usuario; el gateway filtra la respuesta del proveedor para no exponer documentos ajenos.
- Las operaciones de recursos se hacen en `/api/felo/livedocs/resources`, `/readme` o `/tasks`, según la acción y el método HTTP: GET, POST, PUT, PATCH o DELETE. Se admiten carga de archivos de hasta 10 MiB; los archivos originales descargados también se limitan a 10 MiB.

No compartas ni publiques `doc_ref`. Está ligado al UID de Firebase y no autoriza a otro usuario. Al rotar `RASA_AUTH_TOKEN`, las referencias existentes dejan de validarse.

## Pruebas

Las pruebas automatizadas usan respuestas simuladas: no llaman a Felo ni consumen créditos. Probar las funciones desde OmniStudio requiere una sesión autenticada real de Firebase y la app apuntando al gateway desplegado.

## Conectores Google Workspace de Groq

La ruta `POST /api/rasa/groq/workspace` acepta únicamente los conectores de solo lectura Gmail, Google Calendar y Google Drive que el usuario elige para esa consulta. Android obtiene un token de acceso corto mediante Google Identity Services y lo envía en memoria junto con la petición autenticada por Firebase. El gateway lo reenvía a Groq, no lo registra ni lo guarda. No se debe enviar ningún token si el usuario no seleccionó ese conector y pidió una consulta que lo requiera.

Scopes requeridos por los conectores de Groq:

- Gmail: `https://www.googleapis.com/auth/gmail.readonly`
- Calendar: OmniStudio solicita `https://www.googleapis.com/auth/calendar.events.readonly` para limitarse a lectura. La documentación actual de Groq lista `calendar.events`; verifica que el conector acepte el scope más limitado. Si no lo acepta, deja Calendar desactivado en vez de ampliar permisos sin aprobación explícita.
- Drive: `https://www.googleapis.com/auth/drive.readonly`

Configura por separado la pantalla de consentimiento y el cliente Android de Google OAuth para el paquete `com.aistudio.omnistudio.wkspea`, con las huellas SHA-1 de debug y Play App Signing. Para pruebas, añade las cuentas autorizadas como usuarios de prueba. Gmail y Drive requieren verificación de Google para ofrecerlos a usuarios externos en producción. No incluyas tokens de usuario, client secrets ni claves privadas en Android o GitHub. Los permisos se piden solo cuando el usuario conecta/selecciona un servicio; la app no accede en segundo plano.
