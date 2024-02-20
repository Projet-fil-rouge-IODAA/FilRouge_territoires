import streamlit as st
from streamlit_image_comparison import image_comparison
from streamlit_cropper import st_cropper
from PIL import Image

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
    col01, col02 = st.columns([0.8, 0.2], gap='medium')
    with col01:
        st.title("Caractérisation de l'évolution du territoire 🗺️")
    with col02:
        st.image("../app/src/Logo_AgroParisTech.png", width=200)

    st.header("Un approache par clustering de series temporelles.")
    st.subheader("Projet fil Rouge - IODAA 2024.")
    st.markdown("**Auteurs**: Afonso Ponce, Julian Agudelo, \
                Matthieu Verlynde, \
                Nada Kassara, Malek Baroudi.")
    # Introduction
    st.header("Introduction")
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
    st.header("Utilisez les modèles")
    col1, col2 = st.columns([1.1, 1], gap='small')

    with col1:
        st.subheader("Choisissez un région d'interet")
        img = Image.open("../results/crop_image.png")
        cropped_img = st_cropper(img, realtime_update=True, box_color='#0000FF',
                                 aspect_ratio=None)

        col11, col12 = st.columns(2, gap = 'small')
        with col11:
            _ = cropped_img.thumbnail((400, 200))
            st.image(cropped_img)
        with col12:
            st.write("")
            st.write(f"Taille originale: {img.size[0]}px, {img.size[1]}px")
            st.write(f"Taille de votre selection: {cropped_img.size[0]}px, {cropped_img.size[1]}px")
            st.write(f"Nombre de pixels dans votre selection: {cropped_img.size[0]*cropped_img.size[1]}")

    with col2:
        st.subheader("Choisissez le nombre de clusters et la méthodologie")
        n_clust = st.selectbox('Sélectionnez le nombre de clusters: ',
                            ["Automatique", "Personnalisé"])
        if n_clust == "Personnalisé":
            st.slider("", min_value=0, max_value=40)

        st.selectbox('Sélectionnez la méthodologie à utiliser: ',
                    ["Time2feat", "Combinaison de clusterings"])

        executer = st.button("Exécuter")

    if executer:
        st.subheader("Visualisez et analysez les résultats")

        # Visualisation des résultats
        image_comparison(
        img1="../results/just_a_test_base.png",
        img2="../results/just_a_test.png",
        label1="Image originale",
        label2="Image clusterisée",
        width=1000,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True,
        )


if __name__ == "__main__":
    main()
