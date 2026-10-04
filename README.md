# Semantic parser: metas para CLIPS y extracción de entidades

## Propósito y alcance

El flujo actual utiliza Qwen como intérprete de comandos: recibe lenguaje
natural y devuelve metas JSON que el adaptador `goals_to_clips_node` transforma
en hechos `gpsr-goal`. CLIPS expande esos hechos en un plan de acciones.
`semantic_parser` no ejecuta las acciones ni inserta hechos en CLIPS por sí
mismo; un cliente coordina ambos servicios.

El paquete también incluye `entity_extractor_node`, útil durante la ejecución
del plan para reconocer nombres, confirmaciones y otros datos en respuestas
habladas. Este extractor utiliza catálogos YAML y expresiones regulares;
no consulta Qwen ni carga spaCy.

La documentación del motor y sus reglas está en el repositorio
[CLIPS de Biorrobótica UNAM](https://github.com/Biorobotica-UNAM/clips).
Este documento describe el flujo tras la limpieza. El intérprete Qwen está
separado del extractor, cuyos requisitos pendientes se detallan abajo.

## Flujo comando → metas → plan

```text
Comando en lenguaje natural
  ↓ Parser.Request.command
/planning/semantic_parser/semantic_parser
  ↓ qwen_semantic_node → QwenHTTPClient
POST <qwen_url>/translate
  ↓ respuesta HTTP con result.prediction.goals
Parser.Response.response = JSON {input, goals}
  ↓ cliente coordinador: qwen_to_plan.py o API Planning
/planning/goals_to_clips/convert
  ↓ goals_to_clips_node: parseo, normalización y clasificación
gpsr-goal → clips/assert_facts_batch → clips/start_planning
  ↓ goals_planning.clp
/planning/clips/results: plan completo
```

Una respuesta ilustrativa del parser es:

```json
{
  "input": "Go to the kitchen and find the waving person",
  "goals": ["go(kitchen)", "find(person, gesture=waving)"]
}
```

Las metas son strings con la forma `acción(target, clave=valor, ...)`.
El servidor Qwen decide qué metas producir. El cliente HTTP comprueba que
`goals` sea una lista de strings, elimina elementos vacíos y recorta espacios;
no valida si cada acción está soportada por CLIPS. La interpretación de sus
argumentos y la expansión a acciones corresponden al adaptador y a las reglas.

`planning_tools/scripts/qwen_to_plan.py` llama primero al parser, comprueba
que exista una lista `goals` en el nivel superior y envía el JSON completo al
conversor. Después espera el plan en el topic de resultados.
La API C++ ofrece `Planning::parseSemanticCommand()` y
`Planning::goalsToClips()` para coordinar esas mismas etapas. Una llamada
al parser por sí sola no genera un plan.

## Componentes del flujo actual

| Archivo | Responsabilidad |
|---|---|
| `qwen_semantic_node.py` | Servicios ROS para interpretar texto o escuchar un comando GPSR |
| `qwen_http_client.py` | Petición HTTP, autenticación y validación estructural de la respuesta |
| `entity_extractor_node.py` | Servicio ROS de extracción por categoría |
| `extractor.py` | Carga de YAML y búsqueda de términos conocidos |
| `launch/semantic_parser.launch.xml` | Inicio de Qwen y, opcionalmente, del extractor bajo el namespace `semantic_parser` |

`setup.py` registra únicamente `qwen_semantic_node` y `entity_extractor_node`.
El parser spaCy, el comparador y la conversión a CD fueron retirados.

## Interfaces y namespaces

Los servicios se crean con nombres relativos. El launch del paquete añade
`/semantic_parser`; al incluirlo desde `planning_tools`, que añade `/planning`,
se obtienen los siguientes nombres:

| Servicio integrado | Tipo | Uso |
|---|---|---|
| `/planning/semantic_parser/semantic_parser` | `semantic_parser_interfaces/srv/Parser` | Texto → JSON de metas |
| `/planning/semantic_parser/listen_gpsr` | `semantic_parser_interfaces/srv/Parser` | Escucha un comando y devuelve metas |
| `/planning/semantic_parser/extract_entity` | `semantic_parser_interfaces/srv/ExtractEntity` | Extrae valores de una categoría; la interfaz falta en este checkout |

`Parser.srv` sí existe y define:

```text
string command
---
string response
```

La respuesta es un string que contiene JSON, no una estructura ROS con un
campo `goals`. El servicio de conversión de CLIPS recibe ese string en `data`
con el tipo `clips_interfaces/srv/String`.

Ejecutado directamente, Qwen usa `service_name=semantic_parser_qwen`.
El launch lo cambia a `semantic_parser`. El nombre del nodo ROS no determina
el nombre del servicio.

## Comunicación con el servidor Qwen

`QwenHTTPClient.translate()` envía:

```text
POST <qwen_url>/translate
Content-Type: application/json
Authorization: Bearer <qwen_api_key>  # solo si hay clave
```

```json
{"command": "Go to the kitchen"}
```

La forma esperada de la respuesta HTTP es:

```json
{
  "ok": true,
  "result": {
    "normalized_input": "Go to the kitchen",
    "prediction": {"goals": ["go(kitchen)"]}
  },
  "elapsed_ms": 120
}
```

El cliente requiere un objeto con `ok` verdadero y objetos `result` y
`prediction`. Si falta `goals`, utiliza una lista vacía. `normalized_input`
puede omitirse y se sustituye por el comando original. `elapsed_ms` es opcional;
el nodo mide también el tiempo local de la petición para sus logs.

### Parámetros de Qwen

| Parámetro del nodo | Valor directo predeterminado | Función |
|---|---|---|
| `service_name` | `semantic_parser_qwen` | Servicio de interpretación de texto |
| `qwen_url` | `QWEN_SERVER_URL` o `http://127.0.0.1:8008` | URL base del servidor |
| `qwen_api_key` | `QWEN_API_KEY` o vacío | Clave Bearer opcional |
| `timeout_sec` | `QWEN_TIMEOUT_SEC` o `15.0` | Timeout HTTP |
| `response_format` | `QWEN_RESPONSE_FORMAT` o `goals` | Solo se admite `goals`; otros valores impiden iniciar el nodo |
| `gpsr_listen_service_name` | `listen_gpsr` | Servicio de escucha GPSR |
| `listener_service` | `/hri/listener/listen` | Servicio externo de reconocimiento de voz |
| `gpsr_listener_grammar` | vacío | Gramática enviada al listener |
| `gpsr_listen_timeout_sec` | `30.0` | Tiempo de espera de la respuesta del listener |

Una URL o clave vacía pasada como parámetro vuelve a consultar la variable
correspondiente del entorno. El launch del paquete pasa timeout `15.0` y el
launch de integración pasa `5.0`; esos valores explícitos sustituyen el
predeterminado del nodo. Para CLIPS debe utilizarse `response_format=goals`.
El modo `cd` fue retirado junto con su adaptador; el nodo rechaza
explícitamente cualquier `response_format` distinto de `goals`.

### Errores y concurrencia

Un comando de texto vacío devuelve `[]` sin consultar Qwen. Un fallo HTTP,
JSON inválido o respuesta rechazada devuelve un JSON como:

```json
{"status": "aborted", "input": "comando original", "reason": "detalle del error"}
```

El cliente coordinador debe comprobar que la respuesta contenga metas antes
de pedir planificación; la existencia de una respuesta ROS no implica éxito.

El nodo usa `ReentrantCallbackGroup` y `MultiThreadedExecutor` con dos hilos.
Esto permite procesar la respuesta del listener mientras el callback GPSR
espera. La petición HTTP es síncrona y bloquea el hilo de su callback hasta
obtener respuesta o alcanzar el timeout.

## Escucha de un comando GPSR

`listen_gpsr` ignora `request.command` y hace lo siguiente:

1. Espera hasta un segundo a que exista `/hri/listener/listen`.
2. Envía `Listen.Request` con `enable=true` y la gramática configurada.
3. Espera hasta `gpsr_listen_timeout_sec`, consultando el future cada 50 ms.
4. Exige `success=true` y `recognized_speech` no vacío.
5. Envía el texto reconocido a Qwen y devuelve `{input, goals}`.

Este servicio, igual que el servicio de texto, devuelve metas al traducir. Si no se escucha un comando,
devuelve `status=aborted` y una razón. El timeout de escucha y el HTTP son
etapas sucesivas; el cliente ROS debe conceder tiempo para ambas.
`Planning::listenGpsrCommand()` es el cliente C++ de este servicio y tampoco
inicia CLIPS automáticamente.

## Extracción de entidades durante la ejecución

`entity_extractor_node` complementa el flujo de CLIPS en acciones `listen`:

```text
Acción listen del plan
  ↓ executor_tools/Listen.cpp
hri_tools: escucha una respuesta hablada
  ↓ Planning::extractEntity(sentence, category)
/planning/semantic_parser/extract_entity
  ↓ SentenceEntityExtractor + catálogos YAML
value o values
  ↓ executor
resultado de la acción y, en modo completo, actualización de conocimiento
```

No es un paso obligatorio entre Qwen y CLIPS: se utiliza después, cuando
una acción necesita interpretar una respuesta. El executor dispone de modo
simple para devolver el dato y modo completo para actualizar atributos de
personas. Por ejemplo, un atributo `name` se consulta como categoría `NAME`.

### Contrato esperado

El código Python y el cliente C++ esperan una interfaz `ExtractEntity` con:

```text
string sentence
string category
---
string value
string[] values
```

Esta definición describe lo que consume el código; **el archivo
`ExtractEntity.srv` no está presente** en
`Interfaces/Planning/semantic_parser_interfaces`, cuyo CMake solo genera
`Parser.srv`. Hasta resolverlo, el extractor no puede importarse y
`planning_tools` no puede compilar su cliente.

### Diccionarios y algoritmo

El nodo resuelve el directorio:

```text
<share del paquete knowledge_bank>/dictionaries/*.yaml
```

Cada archivo debe tener una categoría y patrones, por ejemplo:

```yaml
label: NAME
patterns:
  - angel
  - robin
```

El catálogo se carga una vez al iniciar; los cambios requieren reiniciar el
nodo. Las categorías se convierten a mayúsculas y los patrones a minúsculas,
reemplazando `_` por espacios. La búsqueda:

1. Convierte la frase a minúsculas y recorta espacios.
2. Busca únicamente en la categoría solicitada.
3. Ordena los patrones por longitud descendente.
4. Busca palabras completas con límites `\b` y escapa caracteres especiales.
5. Borra de una copia de trabajo las coincidencias para evitar que un término
   corto vuelva a coincidir dentro de uno compuesto.
6. Devuelve cada patrón encontrado una vez, normalizado con guiones bajos.

`NAME`, `FOLLOW` y `CONFIRM` devuelven el primer resultado en `value`.
Las demás categorías devuelven la lista en `values`. El orden corresponde a
la búsqueda por longitud, no al orden de aparición en la frase; los empates
no tienen un orden garantizado porque el vocabulario se almacena en sets.
No se infieren sinónimos ni entidades ausentes de los YAML.

Una categoría desconocida o sin coincidencias devuelve `value` vacío y
`values` vacío. Una petición sin frase o categoría devuelve `value` vacío.
No existe aquí un catálogo fijo de categorías: depende de los archivos YAML.

En este checkout `knowledge_bank` contiene otros directorios de datos, pero
no `dictionaries/`. Si el paquete no está registrado en el entorno, la
resolución de su share falla; si existe el share pero faltan los YAML, se
construye un catálogo vacío. Las dependencias `knowledge_bank`, `ament_index_python` y PyYAML ya se
declaran en el paquete. Aun así, la interfaz y los catálogos deben
resolverse para desplegar el extractor.

## Ejecución y prueba del flujo de metas

El flujo Qwen requiere ROS 2, las interfaces `Parser` y `Listen`, `rclpy` y
`httpx`, y un servidor accesible con el contrato `/translate`. La extracción
requiere además `ExtractEntity`, PyYAML y los catálogos de `knowledge_bank`.
spaCy y `en_core_web_trf` ya no son necesarios para este paquete.

Desde la raíz del workspace:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select semantic_parser
source install/setup.bash
```

La compilación de un paquete Python no comprueba que sus imports ROS y sus
recursos estén disponibles durante la ejecución.

Para iniciar únicamente Qwen con el servicio que espera el cliente de CLIPS:

```bash
ros2 run semantic_parser qwen_semantic_node --ros-args \
  -r __ns:=/planning/semantic_parser \
  -p service_name:=semantic_parser \
  -p response_format:=goals \
  -p qwen_url:="http://127.0.0.1:8008"
```

La clave puede proporcionarse mediante `QWEN_API_KEY`. En otras terminales,
con el mismo entorno cargado, iniciar CLIPS y su conversor:

```bash
ros2 run clips_node clips_node --ros-args \
  -r __ns:=/planning -p plan_name:=goals
ros2 run clips_node goals_to_clips_node
```

Los dos comandos anteriores son procesos independientes y requieren sus
propias terminales. Probar el ciclo completo:

```bash
python3 planning_tools/scripts/qwen_to_plan.py \
  "Go to the kitchen and find the waving person"
```

Para consultar solo las metas:

```bash
ros2 service call /planning/semantic_parser/semantic_parser \
  semantic_parser_interfaces/srv/Parser \
  "{command: 'Go to the kitchen and find the waving person'}"
```

Para escuchar un comando, con el listener disponible:

```bash
ros2 service call /planning/semantic_parser/listen_gpsr \
  semantic_parser_interfaces/srv/Parser "{command: ''}"
```

Tras resolver la interfaz y los catálogos del extractor:

```bash
ros2 run semantic_parser entity_extractor_node --ros-args \
  -r __ns:=/planning/semantic_parser

ros2 service call /planning/semantic_parser/extract_entity \
  semantic_parser_interfaces/srv/ExtractEntity \
  "{sentence: 'My name is Angel', category: 'NAME'}"
```

Con el YAML ilustrativo anterior se obtiene `value: angel` y `values: []`.

## Launch del flujo actual

El launch del paquete inicia Qwen como único proveedor del servicio de metas.
Declara `qwen_response_format=goals` y aplica `use_entity_extractor` mediante
una condición `if`. El argumento antiguo `use_spacy_parser` fue eliminado.

El launch de `planning_tools` inicia CLIPS con `plan_name=goals` por defecto y
transmite los parámetros de Qwen. Mientras falten la interfaz `ExtractEntity`
y los catálogos, probar el flujo de metas sin iniciar el extractor:

```bash
ros2 launch planning_tools planning.launch.xml use_entity_extractor:=false
```

También puede iniciarse solo el parser Qwen:

```bash
ros2 launch semantic_parser semantic_parser.launch.xml use_entity_extractor:=false
```

En este segundo caso los servicios están bajo `/semantic_parser`, sin el
prefijo `/planning`; no se inicia CLIPS. Para integrar todo el robot, resolver
los requisitos del extractor y usar `use_entity_extractor:=true`, que es el
valor predeterminado en ambos launches.

El launch no inicia `plan_manager_node` ni el executor. La entrega y ejecución
de instrucciones requieren esos componentes en el despliegue correspondiente.
Compilar `planning_tools` sigue requiriendo la interfaz `ExtractEntity`, aunque
se desactive el nodo extractor en el launch.

## Limpieza realizada

Se retiraron los siguientes componentes ajenos al flujo de metas y al extractor:

| Componentes eliminados | Función anterior |
|---|---|
| `semantic_node.py` y su ejecutable | Parser spaCy de primitivas CD; también se retiró del launch para evitar el servicio duplicado |
| `semantic_compare_node.py`, `semantic_score.py` y el ejecutable del comparador | Evaluación y selección de respuestas CD |
| `debug_pipeline.py` | Depuración manual del parser spaCy |
| `nlp/`, `pipeline/`, `primitives/` | Análisis sintáctico, transformaciones y dataclasses del pipeline CD |
| `verb_to_cd.json` | Diccionario de verbos a primitivas, junto con su instalación como recurso |
| `qwen_goal_adapter.py`, `domain/` | Adaptador Qwen→CD y catálogos usados exclusivamente por los componentes retirados |

Se eliminaron los imports asociados y el modo `response_format=cd`.
Los cuatro programas conservados incluyen comentarios sobre su responsabilidad,
los contratos entre servicios, la espera concurrente del listener y la carga
de catálogos. El extractor resuelve el directorio de conocimiento al construir
el nodo, después de inicializar ROS, en lugar de hacerlo al importar el módulo.

No se añadieron catálogos inventados ni se modificó la interfaz ROS en esta
limpieza. Siguen pendientes `ExtractEntity.srv` y los YAML de `dictionaries/`
para habilitar la extracción durante las acciones `listen`.

Después de recompilar, renovar el entorno con `source install/setup.bash`.
Los directorios anteriores de `build/` e `install/` pueden conservar copias de
los ejecutables retirados: la limpieza de fuentes no elimina esos artefactos.
