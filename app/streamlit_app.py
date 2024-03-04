import streamlit as st
from streamlit_image_comparison import image_comparison
from streamlit_cropper import st_cropper
from function import *
from PIL import Image

st.set_page_config(
    page_title="Evolution de territoires",
    page_icon="🗺️",
    layout="wide",
)

limit_px = 1000

def main():
    """
    The main function
    """
    # Main page
    col01, col02 = st.columns([0.8, 0.2], gap='medium')
    with col01:
        st.title("Caractérisation de l'évolution du territoire 🗺️")
    with col02:
        st.image("app/src/Logo_AgroParisTech.png", width=200)

    st.header("Un approache par clustering de series temporelles.")
    st.subheader("Projet fil Rouge - IODAA 2024.")
    st.markdown("**Auteurs**: Julian Agudelo, Malek Baroudi, \
                Nada Kassara, \
                Matthieu Verlynde, Afonso Ponce.")
    # Introduction
    st.header("Introduction")
    st.write("Cette application est un moyen simple de \
             visualiser et d'utiliser \
             les deux principales méthodologies qui ont été \
             développées au cours de ce projet:")
    st.markdown(f"Un Système d'apprentissage de représentations pour le \
                clustering de séries temporelles multivariées basé sur \
                [Time2feat](https://www.vldb.org/pvldb/vol16/p193-tiano.pdf).\
                \n\
                Un système de combinaison de clusterings basé sur \
                [Une approche de combinaison de clusterings basée sur le vote itératif](https://www.sciencedirect.com/science/article/pii/S131915781930597X).")
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
        img = Image.open("app/src/crop_image.png")
        cropped_img, box = st_cropper(img, realtime_update=True, box_color='#0000FF',
                                      aspect_ratio=None, return_type='both')
        coords = (box['left'], box['top'],
                  box['left']+box['width'],
                  box['top']+box['height'])

        col11, col12 = st.columns(2, gap='small')
        with col11:
            _ = cropped_img.thumbnail((400, 200))
            st.image(cropped_img)
            # st.write(coords)
        with col12:
            st.write("")
            st.write(f"Taille originale: {img.size[0]}px, {img.size[1]}px")
            st.write(f"Taille de votre selection: {cropped_img.size[0]}px, {cropped_img.size[1]}px")
            st.write(f"Nombre de pixels dans votre selection: {cropped_img.size[0]*cropped_img.size[1]}")

    with col2:
        st.subheader("Choisissez la méthode puis le nombre de clusters")

        methode = st.selectbox('Sélectionnez la méthodologie à utiliser: ',
                    ["Combinaison de clusterings","Time2feat"])
        if methode == "Time2feat":
            t2f_methode = st.selectbox('Sélectionnez la méthode de clustering à utiliser: ',
                                       ["Hierarchique", "Kmeans"])
            t2f_transform = st.selectbox('Sélectionnez la méthode de transformation: ',
                                       ["Minmax", "Normalisation standard"])
            n_cores = st.slider("Nombre de coeurs à utiliser:", min_value=1, max_value=30)
            n_clust = st.selectbox('Sélectionnez le nombre de clusters: ',
                                   ["Automatique", "Personnalisé"])
            if n_clust == "Personnalisé":
                 n_clust = st.slider("", min_value=1, max_value=30)
        elif methode == "Combinaison de clusterings":
             vote_methode = st.selectbox('Sélectionnez la méthode à utiliser: ',
                                         ["UMap-HDBSCAN","DTW-Kmedoids"])
             n_clust = st.slider("Sélectionnez le nombre de clusters:", min_value=1, max_value=30)
             

        demo = st.checkbox("Mode démo")

        executer = st.button("Exécuter")

        if cropped_img.size[0]*cropped_img.size[1] > limit_px:
                st.warning("La taille de votre selection est trop grande. \
                        Veuillez réduire la taille de votre selection.")

    if executer:
        st.subheader("Visualisez et analysez les résultats")
        img1="app/src/1000_pixels_422_399_base.png"
        img2="app/src/1000_pixels_422_399_t2f_KMeans_std_demo.png"
        if not demo and cropped_img.size[0]*cropped_img.size[1] > limit_px:
                st.error("La taille de votre selection est trop grande. \
                        Veuillez réduire la taille de votre selection. \
                        \n\
                        Mode démonstration activé.")
                demo = True
        if demo:
            methode, n_clust = "Time2feat - Kmeans - Normalisation standard", 3
        else:
            with st.spinner('Veuillez patienter...'):
                if methode == "Time2feat":
                    t2f_methode = ("Hierarchical" if t2f_methode == "Hierarchique" else "Kmeans")
                    t2f_transform = ("minmax" if t2f_transform == "Minmax" else "std")
                    if n_clust == "Automatique":
                        n_clust = 'auto'
                    link=t2f_apply(coords=coords, model_type=t2f_methode, transform_type=t2f_transform, number_clusters=n_clust, nombre_coeurs=n_cores)
                    img1=link+"_base.png"
                    img2=link+"_no_trans.png"

                    # img2="../results/just_a_test.png"
                elif methode == "Combinaison de clusterings":
                    link=collabclust_apply(coords=coords, model_type=vote_methode, number_clusters=n_clust)
                    img1=link+"_base.png"
                    img2=link+"_no_trans.png"
                time.sleep(5)


        # Visualisation des résultats
        print(img1)
        print(img2)

        image_comparison(
            img1=img1,
            img2=img2,
            label1="Image originale",
            label2="Image clusterisée",
            width=1000,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True,
            )
        
        st.markdown("**Méthode**: "+methode+" \
                    \n\
                    **Nombre de clusters**: "+str(n_clust))


if __name__ == "__main__":
    main()
