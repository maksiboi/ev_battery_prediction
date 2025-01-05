from app import create_app

# Kreiranje i pokretanje aplikacije
app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
