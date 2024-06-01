var searchInput = document.querySelector("#searchInput")
var searchButton = document.querySelector("#searchButton")


function search() {
    const keyword = document.querySelector("#searchInput").value;
    const page = 1; // Start from the first page for new search
    const url = `/search_categories/${encodeURIComponent(keyword)}/${page}`;
    window.location.href = url;
}
