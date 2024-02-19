import streamlit as st

st.set_page_config(
    page_title="Evolution de territoires",
    page_icon="🗺️",
    layout="wide",
)

def main():
    """
    The main function
    """
    # Main page
    st.title("Caractérisation de l'évolution du territoire 🗺️")
    st.header("Un approache par clustering de series temporelles.")
    st.subheader("Projet fil Rouge - IODAA 2024.")
    st.markdown("**Auteurs**: Afonso Ponce, Julian Agudelo, \
                Matthieu Verlynde, \
                Nada Kassara, Malek Baroudi.")
    # Introduction
    st.subheader("Introduction")
    st.write("Cette application est un moyen simple de \
             visualiser et d'utiliser \
             les deux principales méthodologies qui ont été \
             développées au cours de ce projet:")
    st.markdown(f"Un Systéme d'apprentissage de représentations pour le \
                clustering de séries temporelles multivariées basé sur \
                [Time2feat](https://www.vldb.org/pvldb/vol16/p193-tiano.pdf).\
                \n\
                Un système de combinaison de clusterings basé sur \
                [Une approche de combinaison de clusterings basée sur le vote itératif] \
                (https://www.sciencedirect.com/science/article/pii/S131915781930597X).")
    st.write("Grâce à ces deux approches, l'utilisateur peut obtenir \
             différentes propositions de groupes de pixels à partir de \
             l'image satellite en fonction de l'évolution temporelle.\
             \n \
             En utilisant cette application, l'utilisateur peut choisir \
             la zone sûr laquelle il souhaite exécuter les algorithmes \
             de clustering ainsi que le nombre de clusters \
             qu'il souhaite générer.")

    # Utilisez les modèles
    st.subheader("Utilisez les modèles")
    st.slider("Sélectionnez le nombre de clusters: ", min_value=0, max_value=40)
    st.selectbox('Sélectionnez la méthodologie à utiliser: ',
                 ["Time2feat", "Combinaison de clusterings"])
    st.button("Lancer")


if __name__ == "__main__":
    main()
