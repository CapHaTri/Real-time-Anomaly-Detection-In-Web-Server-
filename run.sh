docker cp pinot/def-table.json pinot:/opt/pinot
docker cp pinot/def-schema.json pinot:/opt/pinot
docker exec pinot bash -c "/opt/pinot/bin/pinot-admin.sh AddTable -tableConfigFile /opt/pinot/def-table.json -schemaFile /opt/pinot/def-schema.json -exec"

