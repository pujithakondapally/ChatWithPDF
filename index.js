const skillIcon = document.getElementsByClassName("tooltip");

const addEventListenersToSkillIcon = () => {
    for (let i = 0; i < skillIcon.length; i++) {
        skillIcon[i].addEventListener("click", (e) => {
            console.log(e.target)
            updateDescription(e.target.id);
        });
    }
};

addEventListenersToSkillIcon();

const descriptionDiv = document.getElementsByClassName("about-section");
const updateDescription = (id) => {
    let description = "";

    switch (id) {
        case "googleColab":
            description = "Google Colab: User-friendly platform for code writing, execution, and sharing. Beloved by AI experts, students, developers, and researchers for data analysis, ML, and AI exploration.";
            break;
        case "gradio":
            description = "Gradio: A user-friendly tool that allows to create and share interactive Al Apps without extensive Coding Knowledge.";
            break;
        case "google-ai":
            description = "Google's AI Studio: Google's AI Studio is a platform that allows developers to build, train, and deploy machine learning models using Google's AI and machine learning tools and infrastructure. It simplifies the process of creating AI solutions. ";
            break;
        case "huggingFace":
            description = "HuggingFace: The ultimate destination for building, training, and deploying cutting-edge machine learning models! Revolutionize your AI projects with state-of-the-art NLP and more!";
            break;
        case "langChain":
            description = "LangChain: Seamlessly combine Large Language Models (LLMs) with external computation/data. Build chatbots, analyze data effortlessly. Open source for contributions.";
            break;
        default:
            description = "Description of the selected icon will appear here.";
    }
    for (let i = 0; i < descriptionDiv.length; i++) {
        descriptionDiv[i].textContent = description;
    }
};

updateDescription("google-ai");
