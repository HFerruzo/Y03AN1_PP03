import streamlit as st
# Inicialización de variables
usuarios = []

# Función para agregar un usuario
def agregar_usuario(nombre):
    usuarios.append(nombre)
    st.success(f"Usuario {nombre} agregado.")

# Función para mostrar usuarios
def mostrar_usuarios():
    if usuarios:
        st.write("Lista de usuarios:")
        for usuario in usuario:
            st.white(f" - {usuario}")
    else:
        st.warning("No hay usuarios registrados.")

# Menu Principal
st.title("Gestión de Usuarios")
option = st.selectbox("Selecciona una opción", ["Agregar Usuario", "Mostrar Usuario"])
if option == "Agregar Usuario":
    nombre = st.text_input("Nombre del usuario")
    if st.button("Agregar"):
        if nombre:
            agregar_usuario(nombre)
        else:
            st.error("El  nombre no puede estar vacio")
elif opcion == "Mostrar Usuarios":
    mostrar_usuarios()
