# Use the .NET SDK image to build and run
FROM mcr.microsoft.com/dotnet/sdk:7.0

WORKDIR /app

# Copy project file and source code
COPY hello.csproj .
COPY program.cs .

# Build the program
RUN dotnet build -c Release

# Run it
CMD ["dotnet", "run", "-c", "Release"]

