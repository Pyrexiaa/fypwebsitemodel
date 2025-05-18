$resourceGroupName = "rg-flask"
$acrName = "fypflaskmodelnew"
$acrImage = "$acrName.azurecr.io/fypwebsitemodel:v1"
$location="eastus"
$containerAppEnv="fypflaskmodelenv"
$containerAppName="fypflaskmodelapp"

az containerapp env create --name $containerAppEnv --resource-group $resourceGroupName --location $location

az acr update -n $acrName --admin-enabled true

$acrUsername=az acr credential show --name $acrName --query "username" --output tsv
# acrUsername=$(az acr credential show --name $acrName --query "username" --output tsv)

$acrPassword = az acr credential show --name $acrName --query "passwords[0].value" --output tsv
# acrPassword=$(az acr credential show --name $acrName --query "passwords[0].value" --output tsv)

az containerapp create --name $containerAppName --resource-group $resourceGroupName --environment $containerAppEnv --image $acrImage --registry-server "$acrName.azurecr.io" --registry-username $acrUsername --registry-password $acrPassword --target-port 80 --ingress "external" --cpu 0.5 --memory 1.0Gi

$containerAppUrl = az containerapp show --name $containerAppName --resource-group $resourceGroupName --query "properties.configuration.ingress.fqdn" --output tsv
# containerAppUrl=$(az acr credential show --name $acrName --query "passwords[0].value" --output tsv)

Write-Output "The container app is accessible at http://$containerAppUrl"