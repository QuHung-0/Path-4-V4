# ========= FILE NAME: M06_P02_writeOneJsonRecordIntoKnowledgeGraph.py =========
# FILE ROLE: Reads one enriched JSON record and writes Image, Prediction, Species nodes to Neo4j
# TOTAL FUNCTIONS IN THIS FILE: 4

# FUNCTION 4
def P02_F04_mergeImageNodeIntoGraph(session, image_id: str, source_dataset: str) -> None:
    # MERGE creates the Image node only if it does not already exist
    # using the image_id (SHA-256 hash) means the same image is never duplicated
    session.run(
        """
        MERGE (img:Image {imageId: $image_id})
        SET img.sourceDataset = $source_dataset
        """,
        image_id=image_id,
        source_dataset=source_dataset,
    )

# FUNCTION 5
def P02_F05_mergePredictionNodeAndLinkToImageNode(session, record: dict) -> None:
    # creates a Prediction node and links it to its Image node
    # predId uses imageId + model version to stay unique
    pred_id = f"{record['image_id']}_{record['model_version']}"
    session.run(
        """
        MATCH  (img:Image {imageId: $image_id})
        MERGE  (pred:Prediction {predId: $pred_id})
        SET    pred.modelVersion   = $model_version,
               pred.confidence     = $confidence,
               pred.internalLabel  = $internal_label,
               pred.taxonomyStatus = $taxonomy_status,
               pred.createdAt      = $timestamp
        MERGE  (img)-[:HAS_PREDICTION]->(pred)
        """,
        image_id=record["image_id"],
        pred_id=pred_id,
        model_version=record["model_version"],
        confidence=record["confidence"],
        internal_label=record["internal_label"],
        taxonomy_status=record["taxonomy_status"],
        timestamp=record["timestamp"],
    )

# FUNCTION 6
def P02_F06_mergeSpeciesNodeAndLinkToPredictionNode(session, record: dict) -> None:
    # only runs if taxonomy_status is verified — creates Species node and links to Prediction
    if record.get("taxonomy_status") != "verified":
        return

    pred_id = f"{record['image_id']}_{record['model_version']}"
    session.run(
        """
        MATCH  (pred:Prediction {predId: $pred_id})
        MERGE  (s:Species {aphiaId: $aphia_id})
        SET    s.scientificName = $scientific_name,
               s.rank           = 'Species',
               s.authority      = $authority
        MERGE  (pred)-[:PREDICTS]->(s)
        """,                                  # FIX 3: s.authority now SET from record
        pred_id=pred_id,
        aphia_id=record["aphia_id"],
        scientific_name=record["scientific_name"],
        authority=record.get("authority"),    # FIX 3: read authority from record
    )

# FUNCTION 7
def P02_F07_mergeParentTaxonChainFromSpeciesUpToKingdom(session, record: dict) -> None:
    # walks the lineage list and creates PARENT_TAXON relationships up the taxonomy tree
    if record.get("taxonomy_status") != "verified":
        return

    lineage = record.get("lineage", [])
    if not lineage:
        return

    # ensure every taxon node exists
    for taxon in lineage:
        session.run(
            """
            MERGE (t:Species {scientificName: $name})
            SET t.rank = $rank
            """,
            name=taxon["scientificname"],
            rank=taxon["rank"],
        )

    # link each taxon to its parent — lineage goes from Kingdom (index 0) to Species (last)
    for i in range(len(lineage) - 1):
        child_name  = lineage[i + 1]["scientificname"]
        parent_name = lineage[i]["scientificname"]
        session.run(
            """
            MATCH (child:Species  {scientificName: $child_name})
            MATCH (parent:Species {scientificName: $parent_name})
            MERGE (child)-[:PARENT_TAXON]->(parent)
            """,
            child_name=child_name,
            parent_name=parent_name,
        )

    # link the bottom Species node (from F06) to the genus node in the lineage
    aphia_node_name = record["scientific_name"]
    genus_name = lineage[-2]["scientificname"] if len(lineage) >= 2 else None
    if genus_name:
        session.run(
            """
            MATCH (s:Species {scientificName: $species_name})
            MATCH (g:Species {scientificName: $genus_name})
            MERGE (s)-[:PARENT_TAXON]->(g)
            """,
            species_name=aphia_node_name,
            genus_name=genus_name,
        )