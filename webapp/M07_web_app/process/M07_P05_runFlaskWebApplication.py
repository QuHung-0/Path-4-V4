# ========= FILE NAME: M07_P05_runFlaskWebApplication.py =========
# FILE ROLE: Defines all routes and starts the Flask server — this is what you run to launch the app
# TOTAL FUNCTIONS IN THIS FILE: 4 (route handlers)
# FUNCTION INDEX RANGE: P05 routes

# import Flask helpers for templates, requests, sessions, redirects, flash messages
from flask import render_template, request, session, redirect, url_for, flash

# import server configuration and logger
from webapp.M07_web_app.input.M07_I01_appConfig import (
    M07_FLASK_HOST,   # Flask host address
    M07_FLASK_PORT,   # Flask port number
    M07_FLASK_DEBUG,  # Flask debug mode

    LOGGER            # logger
)

# import startup app factory
from webapp.M07_web_app.process.M07_P01_startFlaskAppAndLoadModelOnce import createFlaskApp

# import upload/prediction helpers
from webapp.M07_web_app.process.M07_P02_handleUploadImageAndReturnPrediction import (
    P02_F04_receiveUploadedImageFileAndSaveToTempFolder,          # save uploaded image
    P02_F05_validateImageIsRgbAndMeetsMinimumSizeRequirement,     # validate image
    P02_F06_runPretrainedModelOnImageAndGetLabelAndConfidence,    # predict label/confidence
    P02_F07_storePredictionResultInServerSideSessionNotForm,      # save result in session
)

# import knowledge-graph write helpers
from webapp.M07_web_app.process.M07_P03_handleWritePredictionToKnowledgeGraph import (
    P03_F08_readPredictionFromServerSideSessionNotFromUserForm,    # read stored prediction
    P03_F09_enrichPredictionWithTimestampAndWriteToNeo4j,         # write to graph
    P03_F10_redirectToSearchPageAfterSuccessfulWrite,              # redirect helper
)

# import search helpers
from webapp.M07_web_app.process.M07_P04_handleSearchQueryAndReturnResults import (
    P04_F11_readSearchFiltersFromQueryParameters,                  # read query params
    P04_F12_callParameterisedCypherQueryWithUserFilters,           # run graph query
    P04_F13_renderSearchResultsPageWithQueryResults,               # build template context
)

# create the Flask app immediately so routes can attach to it
app = createFlaskApp()


# route for upload page and upload submission
@app.route("/", methods=["GET", "POST"])
def upload_page():
    # if user opens the page, show the upload form
    if request.method == "GET":
        return render_template("M07_T01_uploadPage.html")

    # POST means the user uploaded a file
    image_path, error = P02_F04_receiveUploadedImageFileAndSaveToTempFolder(request)
    if error:
        flash(error, "error")
        return redirect(url_for("upload_page"))

    # validate image format and size before prediction
    valid, error = P02_F05_validateImageIsRgbAndMeetsMinimumSizeRequirement(image_path)
    if not valid:
        flash(error, "error")
        return redirect(url_for("upload_page"))

    # run model prediction
    label, confidence = P02_F06_runPretrainedModelOnImageAndGetLabelAndConfidence(image_path)

    # store the result in server-side session for later graph write
    result = P02_F07_storePredictionResultInServerSideSessionNotForm(session, image_path, label, confidence)

    # show prediction result page
    return render_template("M07_T02_predictionResultPage.html", result=result)


# route for saving the current prediction to Neo4j
@app.route("/write_to_kg", methods=["POST"])
def write_to_kg():
    # get the last prediction from server-side session
    result = P03_F08_readPredictionFromServerSideSessionNotFromUserForm(session)

    # write the record to Neo4j
    success = P03_F09_enrichPredictionWithTimestampAndWriteToNeo4j(result)

    if success:
        flash("Prediction saved to Knowledge Graph successfully.", "success")
    else:
        flash("Could not save to Knowledge Graph — check Neo4j is running.", "error")

    # send user to the search page after write
    return P03_F10_redirectToSearchPageAfterSuccessfulWrite()


# route for searching the knowledge graph
@app.route("/search")
def search_page():
    # read search filters from URL
    filters = P04_F11_readSearchFiltersFromQueryParameters(request)

    # run Neo4j query using chosen filters
    results = P04_F12_callParameterisedCypherQueryWithUserFilters(filters)

    # prepare data for template rendering
    context = P04_F13_renderSearchResultsPageWithQueryResults(results, filters)

    # render the search results page
    return render_template("M07_T03_searchPage.html", **context)


# MAIN
if __name__ == "__main__":
    # print startup information before running the server
    LOGGER("P05", f"starting Flask app on http://{M07_FLASK_HOST}:{M07_FLASK_PORT}")
    LOGGER("P05", "Neo4j graph: http://localhost:7474/browser/")
    LOGGER("P05", "make sure Neo4j Desktop is running before uploading images")

    # start Flask development server
    app.run(host=M07_FLASK_HOST, port=M07_FLASK_PORT, debug=M07_FLASK_DEBUG)