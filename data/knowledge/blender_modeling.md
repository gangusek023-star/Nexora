# Lekcja: modelowanie 3D w Blenderze (Python bpy)

## Jak działa modelowanie kodem
W Blenderze wszystko można zrobić skryptem Python przez moduł `bpy`. Obiekt (`Object`) ma dane siatki (`Mesh`), położenie (`location`), obrót (`rotation_euler`, w radianach) i skalę (`scale`). Jednostka to metr. Oś Z jest pionowa (w górę). Najpierw dodaj prymityw, potem pobierz go jako `bpy.context.active_object`, nadaj nazwę, zmień wymiary, dodaj modyfikatory i materiał.
```python
import bpy, math
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
obj = bpy.context.active_object
obj.name = "Skrzynia"
obj.scale = (1.5, 1, 0.5)                   # skala w osiach X, Y, Z
obj.rotation_euler = (0, 0, math.radians(30))
```

## Prymitywy
```python
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=32, ring_count=16, location=(3, 0, 1))
bpy.ops.mesh.primitive_ico_sphere_add(radius=1, subdivisions=3)
bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=2, vertices=32, location=(0, 3, 1))
bpy.ops.mesh.primitive_cone_add(radius1=1, radius2=0, depth=2, vertices=32)
bpy.ops.mesh.primitive_torus_add(major_radius=1, minor_radius=0.25)
bpy.ops.mesh.primitive_plane_add(size=10)    # podłoże
bpy.ops.mesh.primitive_monkey_add()          # małpka Suzanne do testów
```
Po każdym `primitive_*_add` nowy obiekt jest aktywny: `obj = bpy.context.active_object`. Wymiary w metrach możesz ustawić wprost: `obj.dimensions = (2, 1, 0.5)`.

## Materiały i kolory
```python
def material(nazwa, kolor, metal=0.0, szorstkosc=0.5):
    mat = bpy.data.materials.get(nazwa) or bpy.data.materials.new(nazwa)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*kolor, 1.0)    # RGB 0-1 + alfa
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Roughness"].default_value = szorstkosc
    mat.diffuse_color = (*kolor, 1.0)                          # kolor w podglądzie (Workbench)
    return mat

obj.data.materials.append(material("Czerwony", (0.8, 0.1, 0.1)))
```
Świecenie: w Blenderze 4 i 5 wejście nazywa się `"Emission Color"` i `"Emission Strength"`. Szkło: `"Transmission Weight"` = 1 i `"Roughness"` = 0.

## Modyfikatory (najważniejsze narzędzie modelowania)
Modyfikatory zmieniają kształt bez niszczenia siatki.
```python
bevel = obj.modifiers.new(name="Bevel", type='BEVEL')       # zaokrąglone krawędzie
bevel.width = 0.05
bevel.segments = 3
sub = obj.modifiers.new(name="Subdivision", type='SUBSURF') # wygładzanie
sub.levels = 2
sub.render_levels = 2
arr = obj.modifiers.new(name="Array", type='ARRAY')          # powielanie w rzędzie
arr.count = 5
arr.relative_offset_displace = (1.2, 0, 0)
mir = obj.modifiers.new(name="Mirror", type='MIRROR')        # symetria względem osi X
mir.use_axis[0] = True
sol = obj.modifiers.new(name="Solidify", type='SOLIDIFY')    # grubość dla płaszczyzn
sol.thickness = 0.1
```
Wygładzone cieniowanie: `bpy.ops.object.shade_smooth()` (działa na zaznaczonym obiekcie).

## Wycinanie otworów (Boolean)
```python
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
sciana = bpy.context.active_object
bpy.ops.mesh.primitive_cylinder_add(radius=0.4, depth=3, location=(0, 0, 1), rotation=(math.radians(90), 0, 0))
otwor = bpy.context.active_object
b = sciana.modifiers.new(name="Otwor", type='BOOLEAN')
b.operation = 'DIFFERENCE'
b.object = otwor
otwor.hide_viewport = True
otwor.hide_render = True
```

## Przykład: stół
```python
import bpy
def kostka(nazwa, wymiary, pozycja, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=pozycja)
    o = bpy.context.active_object
    o.name = nazwa
    o.scale = wymiary
    o.data.materials.append(mat)
    b = o.modifiers.new(name="Bevel", type='BEVEL')
    b.width = 0.01
    b.segments = 2
    return o

drewno = bpy.data.materials.new("Drewno")
drewno.use_nodes = True
drewno.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.45, 0.28, 0.12, 1)
drewno.diffuse_color = (0.45, 0.28, 0.12, 1)
kostka("Blat", (1.6, 0.9, 0.06), (0, 0, 0.75), drewno)
for i, (x, y) in enumerate([(-0.7, -0.38), (0.7, -0.38), (-0.7, 0.38), (0.7, 0.38)]):
    kostka(f"Noga{i + 1}", (0.07, 0.07, 0.72), (x, y, 0.36), drewno)
print("Stół gotowy")
```

## Grupowanie i porządek
Obiekty jednego modelu umieść w kolekcji albo podepnij pod rodzica:
```python
kolekcja = bpy.data.collections.new("Stol")
bpy.context.scene.collection.children.link(kolekcja)
for o in [blat]:
    for c in o.users_collection:
        c.objects.unlink(o)
    kolekcja.objects.link(o)
noga.parent = blat                              # rodzic: przesuwanie blatu przesuwa nogi
```
Łączenie w jeden obiekt: zaznacz obiekty, ustaw aktywny i użyj `bpy.ops.object.join()`:
```python
bpy.ops.object.select_all(action='DESELECT')
for o in czesci:
    o.select_set(True)
bpy.context.view_layer.objects.active = czesci[0]
bpy.ops.object.join()
```

## Własna geometria (bmesh)
```python
import bpy, bmesh
mesh = bpy.data.meshes.new("Piramida")
bm = bmesh.new()
v = [bm.verts.new(p) for p in [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0), (0, 0, 1.5)]]
bm.faces.new(v[0:4][::-1])
for i in range(4):
    bm.faces.new((v[i], v[(i + 1) % 4], v[4]))
bm.to_mesh(mesh)
bm.free()
obj = bpy.data.objects.new("Piramida", mesh)
bpy.context.collection.objects.link(obj)
```

## Usuwanie i czyszczenie sceny
Usuwaj tylko wtedy, gdy użytkownik o to prosi.
```python
obj = bpy.data.objects.get("Skrzynia")
if obj:
    bpy.data.objects.remove(obj, do_unlink=True)
```

## Światło i kamera
```python
swiatlo = bpy.data.lights.new("Slonce", type='SUN')
swiatlo.energy = 3
o = bpy.data.objects.new("Slonce", swiatlo)
bpy.context.collection.objects.link(o)
o.rotation_euler = (math.radians(50), 0, math.radians(30))
bpy.ops.object.camera_add(location=(6, -6, 4))
kamera = bpy.context.active_object
bpy.context.scene.camera = kamera
```

## Eksport do Roblox i innych programów
```python
bpy.ops.export_scene.fbx(filepath="C:/modele/stol.fbx", use_selection=True)   # Roblox: Import 3D
bpy.ops.wm.obj_export(filepath="C:/modele/stol.obj")                         # Blender 4+
bpy.ops.export_scene.gltf(filepath="C:/modele/stol.glb")
```
Roblox ma limit 20 000 trójkątów na jedną siatkę (MeshPart) – zmniejsz poziom Subdivision przed eksportem.

## Zasady dobrego modelu
- Zaczynaj od prostych brył i dokładaj szczegóły (blokowanie kształtu).
- Trzymaj rozsądną skalę: człowiek ~1.8 m, drzwi ~2 m, stół ~0.75 m wysokości.
- Krawędzie lekko fazuj (Bevel) – wyglądają realistyczniej.
- Nadawaj obiektom czytelne nazwy i materiały.
- Postaw obiekt na podłożu: dolna krawędź na Z = 0.
