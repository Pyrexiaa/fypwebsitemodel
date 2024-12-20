$rgname = "rg-flask"
$location = "eastus"
$acrname = "fypflaskmodelnew"

az group create --name $rgname --location $location

az acr create --resource-group $rgname --name $acrname --sku Basic

az acr login --name $acrname

docker tag fypwebsitemodel fypflaskmodelnew.azurecr.io/fypwebsitemodel:v1

docker push fypflaskmodelnew.azurecr.io/fypwebsitemodel:v1
