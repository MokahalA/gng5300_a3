import sqlite3
from datetime import datetime, timedelta
from langchain_core.tools import tool
import pytz
from langchain_core.runnables import RunnableConfig
from typing import Optional, List, Union

''' This is a script that contains 10 LangChain tools to support the AI assistant's capabilities.'''

# Path to the SQLite database file
db = "skincare.sqlite"


@tool
def get_product_categories() -> list[str]:
    """Fetch all product categories from the database.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = """
    SELECT 
        DISTINCT category
    FROM 
        products
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    results = [row[0] for row in rows]

    cursor.close()
    conn.close()

    return results


@tool
def search_product_by_name(product_name: str) -> Union[dict, List[dict]]:
    """Fetch up to 3 products by partial match of their name."""
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        # Use the LIKE operator for partial matching and LIMIT the results to 4
        query = """
        SELECT 
            product_id, product_name, description, category, stock, price
        FROM 
            products
        WHERE 
            product_name LIKE ?
        LIMIT 3
        """
        # Use '%' for partial matching on both sides
        cursor.execute(query, (f"%{product_name}%",))
        rows = cursor.fetchall()

        if not rows:
            return {"message": "No matching products found."}

        # If there are matches, return them as a list of dictionaries
        results = [
            {
                "product_id": row[0],
                "product_name": row[1],
                "description": row[2],
                "category": row[3],
                "stock": row[4],
                "price": row[5]
            }
            for row in rows
        ]
    except Exception as e:
        return {"message": f"Error: {str(e)}"}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    # Return a single product if only one match is found, else return a list
    return results[0] if len(results) == 1 else results


@tool
def get_recommendations(category: str, description: str) -> Union[List[dict], dict]:
    """Get up to 3 product recommendations based on category and partial match in description.
       If no matches are found with the description, fetch products by category only.
    """
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        # Step 1: Try fetching by category and partial match in description
        query_with_description = """
        SELECT 
            product_id, product_name, description, category, stock, price
        FROM 
            products
        WHERE 
            LOWER(category) = LOWER(?) AND LOWER(description) LIKE LOWER(?)
        LIMIT 3
        """
        cursor.execute(query_with_description, (category, f"%{description}%"))
        rows = cursor.fetchall()

        # Step 2: If no results found, fallback to fetching by category only
        if not rows:
            query_fallback = """
            SELECT 
                product_id, product_name, description, category, stock, price
            FROM 
                products
            WHERE 
                LOWER(category) = LOWER(?)
            LIMIT 3
            """
            cursor.execute(query_fallback, (category,))
            rows = cursor.fetchall()

            if not rows:
                return {"message": "No relevant products found in this category."}

        # Prepare results as a list of dictionaries
        results = [
            {
                "product_id": row[0],
                "product_name": row[1],
                "description": row[2],
                "category": row[3],
                "stock": row[4],
                "price": row[5]
            }
            for row in rows
        ]

    except Exception as e:
        return {"message": f"Error: {str(e)}"}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    return results


@tool
def add_to_cart(config: RunnableConfig, product_id: int, quantity: int = 1) -> dict:
    '''Add a product to the user's cart with the specified quantity.'''
    try:
        # Get the user_id from the configuration to identify their cart
        user_id = config.get("configurable", {}).get("thread_id", None)

        if not user_id:
            raise ValueError("No user_id found in the configuration.")
        
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        # Check if the product exists and get its price, stock, and name
        cursor.execute("SELECT product_name, price, stock FROM products WHERE product_id = ?", (product_id,))
        row = cursor.fetchone()

        if not row:
            return {"message": "Product not found."}
        
        product_name, price, stock = row
        if stock < quantity:
            return {"message": "Insufficient stock."}
        
        # Check to see if the product is already in the cart, increase quantity if so.
        cursor.execute("SELECT quantity FROM shopping_carts WHERE user_id = ? AND product_id = ?", (user_id, product_id))
        row = cursor.fetchone()

        if row:
            quantity += row[0]
            cursor.execute("UPDATE shopping_carts SET quantity = ? WHERE user_id = ? AND product_id = ?", (quantity, user_id, product_id))
        else:
            # Insert the product name along with price and quantity into the cart
            cursor.execute("INSERT INTO shopping_carts (user_id, product_id, product_name, price, quantity) VALUES (?, ?, ?, ?, ?)", 
                           (user_id, product_id, product_name, price, quantity))
        
        conn.commit()
        return {"message": "Product added to cart successfully."}
    except Exception as e:
        return {"message": f"Error: {str(e)}"}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@tool
def remove_from_cart(config: RunnableConfig, product_id: int) -> dict:
    '''Remove a product from the user's cart.'''
    try:
        # Get the user_id from the configuration to identify their cart
        user_id = config.get("configurable", {}).get("thread_id", None)

        if not user_id:
            raise ValueError("No user_id found in the configuration.")
        
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        # Check if the product exists in the cart
        cursor.execute("SELECT * FROM shopping_carts WHERE user_id = ? AND product_id = ?", (user_id, product_id))
        row = cursor.fetchone()

        if not row:
            return {"message": "Product not found in cart."}

        # If the product exists, delete it from the cart
        cursor.execute("DELETE FROM shopping_carts WHERE user_id = ? AND product_id = ?", (user_id, product_id))
        
        conn.commit()
        return {"message": "Product removed from cart successfully."}
    except Exception as e:
        return {"message": f"Error: {str(e)}"}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@tool
def view_cart(config: RunnableConfig) -> Union[dict, List[dict]]:
    '''View the user's cart and calculate the total price.'''
    try:
        # Get the user_id from the configuration to identify their cart
        user_id = config.get("configurable", {}).get("thread_id", None)

        if not user_id:
            raise ValueError("No user_id found in the configuration.")
        
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        # Fetch all products in the user's cart
        cursor.execute("SELECT product_id, product_name, price, quantity FROM shopping_carts WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()

        if not rows:
            return {"message": "Cart is empty."}
        
        # Calculate the total price of the cart
        total_price = sum(price * quantity for _, _, price, quantity in rows)

        # Prepare results as a list of dictionaries
        results = [
            {
                "product_id": row[0],
                "product_name": row[1],
                "price": row[2],
                "quantity": row[3]
            }
            for row in rows
        ]

        return {"total_price": total_price, "products": results}
    except Exception as e:
        return {"message": f"Error: {str(e)}"}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@tool
def get_delivery_time() -> dict:
    """Return the estimated delivery time.
    """
    # Get the current date and time
    now = datetime.now(pytz.timezone('US/Eastern'))

    # Add 7 days to the current date for estimated delivery time
    delivery_time = now + timedelta(days=7)
    return {"expected_delivery_time": delivery_time.strftime("%Y-%m-%d %H:%M:%S")}


@tool
def get_returns_policy() -> dict:
    """Return the return policy details.
    """
    return {
        "policy": "Our return policy lasts 30 days. If 30 days have gone by since your purchase, unfortunately we can’t offer you a refund or exchange.",
        "details": "To be eligible for a return, your item must be unused and in the same condition that you received it. It must also be in the original packaging."
    }

@tool
def get_shipping_policy() -> dict:
    """Return the shipping policy details.
    """
    return {
        "policy": "We offer free shipping on all orders over $50. For orders under $50, a flat rate shipping fee of $5 will be applied.",
        "details": "Orders are typically processed within 1-2 business days. Once your order has been shipped, you will receive a tracking number via email."
    }


@tool
def get_payment_methods() -> list[str]:
    """Return a list of available payment methods.
    """
    return ["Apple Pay", "Google Pay", "Debit Card", "Credit Card"]

