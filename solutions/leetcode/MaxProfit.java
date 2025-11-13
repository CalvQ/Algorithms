/**
 * Maximum Profit
 * 
 * Given an array of the price of a certain stock in chronological order,
 * return the maximum profit you could gain from buy and selling the stock
 * exactly once.
 * 
 * Example:
 *      maxProfit({9, 11, 8, 5, 7, 10})
 *      ->  5   //You buy at 5 and sell at 10, net gain is 5
 */

public class MaxProfit{
    public int maxProfit(int[] stockPrices){
        //Variable to track our best profit
        int profit = 0;

        //Variable to track our lowest-price of stock so far
        int minPrice = stockPrices[0];
        
        //Loop through the stock price array
        for(int i=1; i<stockPrices.length; i++){
            //Establish minimum price
            minPrice = (minPrice < stockPrices[i])?minPrice:stockPrices[i];

            //Maximum was either previous or the diff between the new price
            //and the previous minimum
            profit = (profit > (stockPrices[i]-minPrice))?
                profit:stockPrices[i]-minPrice;
        }

        return profit;
    }

    public static void main(String[] args){
        MaxProfit maxProfitInstance = new MaxProfit();

        int[] input = {9,11,8,5,7,10};
        System.out.println(maxProfitInstance.maxProfit(input));
    }
}