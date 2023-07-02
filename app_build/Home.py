import streamlit as st

# Define valid credentials for manager and employee
valid_credentials = {
    "manager": {"username": "manager", "password": "manager123"},
    "employee": {"username": "employee", "password": "employee123"}
}

# Define Streamlit app
def main():
    st.title("Login Example")

    # Sidebar
    st.sidebar.title("Login")
    role = st.sidebar.selectbox("Role", ("manager", "employee"))
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")

    # Check if login button is clicked
    if login_button:
        if username == valid_credentials[role]["username"] and password == valid_credentials[role]["password"]:
            st.success(f"Logged in as {role}")
            if role == "manager":
                # Show manager content
                st.write("Welcome, Manager!")
            else:
                # Show employee content
                st.write("Welcome, Employee!")
        else:
            st.error("Invalid username or password")

# Run the app
if __name__ == "__main__":
    main()