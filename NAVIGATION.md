# 🧭 SmolRouter Navigation Guide

## How to Access the Upstreams Page

The new **Upstream Providers** page can be accessed from multiple locations in the SmolRouter web interface:

### 📊 From Main Dashboard (`/`)

**Method 1: Header Navigation**
- At the top of the page, you'll see navigation links:
  - `📈 Performance` 
  - `🔗 Upstreams` ← **Click here!**

**Method 2: Action Buttons**
- In the "Recent Requests" section, there are action buttons:
  - `📈 Performance` (green button)
  - `🔗 Upstreams` (purple button) ← **Click here!**
  - `🔄 Refresh` (default button)

### 📈 From Performance Page (`/performance`)

**Header Navigation**
- At the top of the page:
  - `← Dashboard`
  - `🔗 Upstreams` ← **Click here!**

### 🔗 From Upstreams Page (`/upstreams`)

**Navigation Bar**
- `📊 Dashboard` - Go back to main page
- `📈 Performance` - View performance analytics  
- `🔗 Upstreams` - Currently active page (highlighted)

## 🗺️ Complete Navigation Flow

```
Main Dashboard (/)
    ↓
    ├── 📈 Performance (/performance)
    │   └── 🔗 Upstreams (/upstreams)
    │       └── ← Dashboard (/)
    │
    └── 🔗 Upstreams (/upstreams)
        ├── 📊 Dashboard (/)
        └── 📈 Performance (/performance)
```

## 🎯 What You'll See on the Upstreams Page

1. **📦 Summary Cards**: Overview statistics
   - Total Providers
   - Healthy Providers  
   - Total Models
   - Cache Entries

2. **🔧 Control Buttons**:
   - `🔄 Refresh` - Update data from providers
   - `🗑️ Clear Cache` - Force cache refresh

3. **🏗️ Provider Cards**: Detailed view of each upstream
   - Health status (🟢 Healthy / 🔴 Unhealthy)
   - Provider type (OLLAMA / OPENAI)
   - Endpoint URL and priority
   - Available models with aliases
   - Model counts and metadata

4. **📊 Cache Information**: Performance metrics
   - TTL settings and hit counts
   - Provider-specific cache stats

## 🚀 Quick Access Tips

- **Bookmark** `/upstreams` for direct access
- Use **keyboard shortcuts** (if your browser supports them)
- The page **auto-refreshes** every 30 seconds
- **Mobile responsive** - works on phones and tablets

## 🔄 Real-time Features

- **Live health monitoring** of all providers
- **Automatic model discovery** with caching
- **Background health checks** every 30 seconds
- **Manual refresh** capability for immediate updates
- **Cache management** with TTL visualization

The upstreams page provides complete visibility into your model provider infrastructure, making it easy to debug issues, monitor performance, and understand your available model inventory across all configured endpoints.